import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import flopy
import numpy as np
from example_utils import boundary_values, component_fields, executable_path

from mf6pqc.utils import get_gwt_model_name


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]


@dataclass
class Config:
    nx: int
    nz: int
    length: float
    height: float
    width: float
    days: float
    dt: float
    save_every: float
    kv_ratio: float
    injection_rate: float
    injection_depth: float
    extraction_depth: float
    screen_conductance: float
    outlet_head: float
    alpha_l: float
    alpha_t: float
    diffusion: float

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.nz <= 0:
            raise ValueError("Grid dimensions must be positive")
        if any(
            not np.isfinite(value) or value <= 0
            for value in (
                self.length,
                self.height,
                self.width,
                self.days,
                self.dt,
                self.save_every,
            )
        ):
            raise ValueError(
                "Lengths, duration, time step, and save interval must be positive and finite"
            )
        if not np.isclose(self.steps * self.dt, self.days):
            raise ValueError("Duration must be an integer number of steps")
        if not self.inlet_layers.size or not self.outlet_layers.size:
            raise ValueError("The grid must resolve both inlet and outlet screens")

    @property
    def shape(self):
        return (self.nz, 1, self.nx)

    @property
    def nxyz(self):
        return self.nx * self.nz

    @property
    def dx(self):
        return self.length / self.nx

    @property
    def dz(self):
        return self.height / self.nz

    @property
    def steps(self):
        return int(round(self.days / self.dt))

    @property
    def cell_volume(self):
        return self.dx * self.dz * self.width

    @property
    def z(self):
        return self.height - (np.arange(self.nz) + 0.5) * self.dz

    @property
    def x(self):
        return (np.arange(self.nx) + 0.5) * self.dx

    @property
    def inlet_layers(self):
        return np.flatnonzero(self.height - self.z < self.injection_depth)

    @property
    def outlet_layers(self):
        return np.flatnonzero(self.height - self.z < self.extraction_depth)

    @property
    def save_steps(self):
        n = max(1, round(self.save_every / self.dt))
        return sorted(set([1, *range(n, self.steps + 1, n), self.steps]))

    def k_field(self):
        source = np.load(CASE_DIR / "input_data/hk.npy")
        if source.shape[0] % self.nz or source.shape[1] % self.nx:
            raise ValueError("Grid must divide the 100 x 400 source field")
        logk = (
            np.log(source)
            .reshape(self.nz, source.shape[0] // self.nz, self.nx, source.shape[1] // self.nx)
            .mean(axis=(1, 3))
        )
        return np.exp(logk).reshape(-1)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ChemistryConfig:
    porosity: float
    mineral_molar_volumes: dict[str, float]
    mineral_volume_fractions: dict[str, float]
    elements: list[str]
    temperature: float
    pressure: float
    density: float

    @property
    def inert_fraction(self) -> float:
        return 1 - self.porosity - sum(self.mineral_volume_fractions.values())

    def to_dict(self) -> dict:
        return asdict(self)

def build_model(
    *,
    workspace: str | Path,
    species: list[str],
    initial_concentrations: np.ndarray,
    inflow_concentrations: np.ndarray,
    config: Config,
    porosity: float,
    hydraulic_conductivity: np.ndarray,
    reference_density: float,
    update_density: bool,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = config.nxyz
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    inflow_concentrations = boundary_values(species, inflow_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    simulation = flopy.mf6.MFSimulation(
        sim_name="channel2d", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, time_units="DAYS", nper=1, perioddata=[(config.days, config.steps, 1.0)]
    )
    discretization = dict(
        nlay=config.nz,
        nrow=1,
        ncol=config.nx,
        delr=config.dx,
        delc=config.width,
        top=config.height,
        botm=config.height - np.arange(1, config.nz + 1) * config.dz,
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        filename="flow.ims",
        print_option="SUMMARY",
        outer_dvclose=1e-08,
        outer_maximum=150,
        inner_maximum=500,
        inner_dvclose=1e-09,
        rcloserecord=1e-07,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        relaxation_factor=0.97,
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(gwf, pname="DIS", **discretization)
    flopy.mf6.ModflowGwfic(gwf, strt=config.outlet_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity.reshape(config.shape),
        k33=(config.kv_ratio * hydraulic_conductivity).reshape(config.shape),
    )
    well = [
        (
            (int(layer), 0, 0),
            config.injection_rate / len(config.inlet_layers),
            *inflow_concentrations,
        )
        for layer in config.inlet_layers
    ]
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="INLET",
        filename="inlet.wel",
        save_flows=True,
        auxiliary=species,
        stress_period_data=well,
    )
    outlet = [
        (
            (int(layer), 0, config.nx - 1),
            config.outlet_head,
            config.screen_conductance / len(config.outlet_layers),
            *(
                concentration[layer * config.nx + config.nx - 1]
                for concentration in initial_fields.values()
            ),
        )
        for layer in config.outlet_layers
    ]
    flopy.mf6.ModflowGwfdrn(
        gwf,
        pname="OUTLET",
        filename="outlet.drn",
        save_flows=True,
        auxiliary=species,
        stress_period_data=outlet,
    )
    records = [("HEAD", "STEPS", *config.save_steps), ("BUDGET", "STEPS", *config.save_steps)]
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord="flow.hds",
        budget_filerecord="flow.bud",
        saverecord=records,
        printrecord=[("BUDGET", "LAST")],
    )
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(simulation, modelname=gwt_name, save_flows=True)
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            filename=f"{gwt_name}.ims",
            print_option="SUMMARY",
            outer_dvclose=1e-08,
            outer_maximum=80,
            inner_dvclose=1e-09,
            inner_maximum=250,
            rcloserecord=1e-08,
            linear_acceleration="BICGSTAB",
            scaling_method="DIAGONAL",
            relaxation_factor=0.97,
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, pname="DIS", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration.reshape(config.shape))
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=config.alpha_l,
            alv=config.alpha_l,
            ath1=config.alpha_t,
            ath2=config.alpha_t,
            atv=config.alpha_t,
            diffc=config.diffusion,
        )
        flopy.mf6.ModflowGwtmst(gwt, pname="MST", porosity=porosity)
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[("INLET", "AUX", species_name), ("OUTLET", "AUX", species_name)]
        )
        flopy.mf6.ModflowGwtoc(
            gwt, concentration_filerecord=f"{gwt_name}.ucn", saverecord=[("CONCENTRATION", "LAST")]
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea="gwf_model",
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )
    if update_density:
        flopy.mf6.ModflowGwfbuy(
            gwf,
            pname="BUY",
            denseref=reference_density,
            nrhospecies=1,
            packagedata=[(0, 0.0, 0.0, get_gwt_model_name("Cl"), "CONCENTRATION")],
        )
    simulation.write_simulation(silent=True)
    return simulation
