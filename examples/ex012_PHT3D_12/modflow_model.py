from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import flopy
import numpy as np
from example_utils import boundary_values, component_fields, executable_path

from mf6pqc.utils import get_gwt_model_name


def build_model(
    *,
    workspace: str | Path,
    species: list[str],
    initial_concentrations: np.ndarray,
    pulse_concentrations: np.ndarray,
    chase_concentrations: np.ndarray,
    nlay: int,
    nrow: int,
    ncol: int,
    delr: float | list[float] | np.ndarray,
    delc: float | list[float] | np.ndarray,
    top: float | list[float] | np.ndarray,
    botm: float | list[float] | np.ndarray,
    period_data: list[tuple[float, int, float]],
    porosity: float,
    hydraulic_conductivity: float,
    initial_head: float,
    outlet_head: float,
    inflow_rate: float,
    alh: float,
    diffc: float,
    pht3d_mobile_components: frozenset[str],
    pulse_end: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nlay * nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    pulse_concentrations = boundary_values(species, pulse_concentrations)
    chase_concentrations = boundary_values(species, chase_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    simulation = flopy.mf6.MFSimulation(
        sim_name="model", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, time_units="DAYS", nper=len(period_data), perioddata=period_data
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="NONE",
        complexity="SIMPLE",
        outer_dvclose=1e-10,
        inner_dvclose=1e-11,
        rcloserecord=1e-10,
        linear_acceleration="CG",
        filename="flow.ims",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfdis(gwf, **discretization)
    flopy.mf6.ModflowGwfic(gwf, strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_flows=True, save_specific_discharge=True, icelltype=0, k=hydraulic_conductivity
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0, steady_state={0: True})
    pulse = [[(0, 0, 0), inflow_rate, *pulse_concentrations]]
    chase = [[(0, 0, 0), inflow_rate, *chase_concentrations]]
    pulse_end_period = (
        next(
            (
                index
                for index, elapsed in enumerate(np.cumsum([entry[0] for entry in period_data]))
                if np.isclose(elapsed, pulse_end)
            )
        )
        + 1
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-INLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={0: pulse, pulse_end_period: chase},
        auxiliary=species,
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-OUTLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={0: [[(0, 0, ncol - 1), outlet_head]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="gwf_model.bud",
        head_filerecord="gwf_model.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        is_mobile = species_name in pht3d_mobile_components
        if not is_mobile and species_name != "Charge":
            raise ValueError(f"Unexpected immobile PHT3D component: {species_name}")
        transport_rclose = 1e-09 if species_name in {"H2O", "H", "O"} else 1e-12
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=False, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="NONE",
            complexity="SIMPLE",
            outer_dvclose=1e-11,
            outer_maximum=200,
            inner_maximum=1000,
            inner_dvclose=1e-12,
            rcloserecord=transport_rclose,
            linear_acceleration="BICGSTAB",
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        if is_mobile:
            flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
            flopy.mf6.ModflowGwtdsp(
                gwt,
                xt3d_off=True,
                alh=alh,
                ath1=0.1 * alh,
                diffc=diffc,
                filename=f"{gwt_name}.dsp",
            )
            flopy.mf6.ModflowGwtssm(
                gwt, sources=[("WEL-INLET", "AUX", species_name)], filename=f"{gwt_name}.ssm"
            )
        else:
            flopy.mf6.ModflowGwtcnc(
                gwt,
                pname="CNC-ZERO-CHARGE",
                maxbound=nxyz,
                stress_period_data={0: [[(0, 0, column), 0.0] for column in range(ncol)]},
                filename=f"{gwt_name}.cnc",
            )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        if is_mobile:
            flopy.mf6.ModflowGwfgwt(
                simulation,
                exgtype="GWF6-GWT6",
                exgmnamea=gwf.name,
                exgmnameb=gwt.name,
                filename=f"{gwt_name}.gwfgwt",
            )
    simulation.write_simulation(silent=True)
    return simulation
