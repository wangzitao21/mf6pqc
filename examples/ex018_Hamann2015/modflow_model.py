import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import flopy
import numpy as np
from example_utils import boundary_values, component_fields, executable_path

from mf6pqc.utils import get_gwt_model_name


@dataclass(frozen=True)
class Grid:
    delr: np.ndarray
    delv: np.ndarray
    top: float

    @property
    def nlay(self) -> int:
        return int(self.delv.size)

    @property
    def nrow(self) -> int:
        return 1

    @property
    def ncol(self) -> int:
        return int(self.delr.size)

    @property
    def nxyz(self) -> int:
        return self.nlay * self.nrow * self.ncol

    @property
    def botm(self) -> np.ndarray:
        return self.top - np.cumsum(self.delv)

    @property
    def x_centres(self) -> np.ndarray:
        return np.cumsum(self.delr) - 0.5 * self.delr

    @property
    def z_centres(self) -> np.ndarray:
        layer_tops = np.r_[self.top, self.botm[:-1]]
        return layer_tops - 0.5 * self.delv


@dataclass(frozen=True)
class TimeConfig:
    perioddata: tuple[tuple[float, int, float], ...]
    snapshot_years: tuple[float, ...]
    snapshot_steps_global: tuple[int, ...]
    snapshot_steps_by_period: dict[int, list[int]]


def _output_records(time_config: TimeConfig, variable: str) -> dict[int, list[tuple]]:
    return {
        kper: [(variable, "STEPS", *steps)]
        for kper, steps in time_config.snapshot_steps_by_period.items()
    }


def build_model(
    *,
    workspace: str | Path,
    species: list[str],
    initial_concentrations: np.ndarray,
    recharge_concentrations: np.ndarray,
    grid: Grid,
    time_config: TimeConfig,
    porosity: float,
    hydraulic_conductivity: float,
    recharge_rate: float,
    alh: float,
    ath1: float,
    diffc: float,
    days_per_year: float,
    evaporation_rates: np.ndarray,
    reference_density: float,
    split_x: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = grid.nxyz
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    recharge_concentrations = boundary_values(species, recharge_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    simulation = flopy.mf6.MFSimulation(
        sim_name="hamann2015", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation,
        pname="tdis",
        time_units="DAYS",
        nper=len(time_config.perioddata),
        perioddata=list(time_config.perioddata),
    )
    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=0.0001,
        outer_maximum=100,
        inner_maximum=500,
        inner_dvclose=1e-06,
        rcloserecord=1e-06,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.97,
        filename="flow.ims",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=grid.nlay,
        nrow=grid.nrow,
        ncol=grid.ncol,
        delr=grid.delr,
        delc=1.0,
        top=grid.top,
        botm=grid.botm,
    )
    flopy.mf6.ModflowGwfdis(gwf, pname="DIS", **discretization)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="IC", strt=grid.top)
    flopy.mf6.ModflowGwfsto(
        gwf, pname="STO", save_flows=True, iconvert=0, ss=0.0, sy=0.0, transient={0: True}
    )
    split_col = int(np.count_nonzero(grid.x_centres < split_x))
    recharge_spd = [((0, 0, j), recharge_rate, *recharge_concentrations) for j in range(split_col)]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="RECHARGE",
        filename="gwf_model_recharge.rch",
        fixed_cell=True,
        auxiliary=species,
        stress_period_data={0: recharge_spd},
    )
    evaporation_aux = np.zeros(len(species), dtype=float)
    evaporation_spd = [
        ((0, 0, j), -(rate / 1000.0) / days_per_year, *evaporation_aux)
        for j, rate in zip(range(split_col, grid.ncol), evaporation_rates, strict=True)
    ]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="EVAPORATION",
        filename="gwf_model_evaporation.rch",
        fixed_cell=True,
        auxiliary=species,
        stress_period_data={0: evaporation_spd},
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="HEAD_REFERENCE",
        auxiliary=species,
        stress_period_data={0: [((0, 0, 0), grid.top, *recharge_concentrations)]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="OC",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord={
            kper: [("HEAD", "STEPS", *steps), ("BUDGET", "STEPS", *steps)]
            for kper, steps in time_config.snapshot_steps_by_period.items()
        },
    )
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, model_nam_file=f"{gwt_name}.nam", save_flows=False
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="SUMMARY",
            outer_dvclose=1e-06,
            outer_maximum=100,
            inner_maximum=200,
            inner_dvclose=1e-07,
            rcloserecord=1e-06,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alh,
            alv=alh,
            ath1=ath1,
            atv=ath1,
            diffc=diffc,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        flopy.mf6.ModflowGwtsrc(
            gwt,
            pname="SRC",
            maxbound=grid.nxyz,
            stress_period_data={
                0: [((k, 0, j), 0.0) for k in range(grid.nlay) for j in range(grid.ncol)]
            },
            filename=f"{gwt_name}.src",
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            pname=f"{species_name}_SSM",
            sources=[
                ("HEAD_REFERENCE", "AUX", species_name),
                ("RECHARGE", "AUX", species_name),
                ("EVAPORATION", "AUX", species_name),
            ],
            filename=f"{gwt_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=_output_records(time_config, "CONCENTRATION"),
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )
    flopy.mf6.ModflowGwfbuy(
        gwf,
        pname="BUY",
        denseref=reference_density,
        nrhospecies=1,
        density_filerecord="model_density.bin",
        packagedata=[(0, 0.0, 0.0, "gwt_Cl_model", "CONCENTRATION")],
    )
    simulation.write_simulation(silent=True)
    return simulation
