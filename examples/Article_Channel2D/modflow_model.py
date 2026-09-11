"""MODFLOW grid, shared physical configuration and Pitzer chemistry."""

import os
import sys
from pathlib import Path

# Keep generated Python bytecode out of the six-entry example layout.
sys.dont_write_bytecode = True
REPOSITORY_DIR = Path(__file__).resolve().parents[2]
if os.environ.get("MF6PQC_USE_INSTALLED") != "1":
    sys.path.insert(0, str(REPOSITORY_DIR))


def library_path(version="mf6.8.0"):
    """Locate the MODFLOW 6.8.0 shared library for this platform."""
    override = os.environ.get("MF6PQC_LIBMF6")
    if override:
        return str(Path(override).expanduser().resolve())
    name = (
        "libmf6.dll"
        if sys.platform == "win32"
        else "libmf6.dylib"
        if sys.platform == "darwin"
        else "libmf6.so"
    )
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    return str((directory / name).expanduser().resolve())


def executable_path(version="mf6.8.0"):
    """Locate the standalone MODFLOW executable used by FloPy."""
    override = os.environ.get("MF6PQC_MF6_EXE")
    if override:
        return str(Path(override).expanduser().resolve())
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    return str((directory / ("mf6.exe" if sys.platform == "win32" else "mf6")).resolve())


def runtime_path(case_file, kind):
    """Keep inputs local and optionally isolate generated model/results files."""
    if kind not in {"output", "simulation"}:
        raise ValueError("kind must be output or simulation")
    case = Path(case_file).resolve().parent
    override = os.environ.get("MF6PQC_RUN_ROOT")
    base = Path(override).expanduser().resolve() / case.name if override else case
    return base / kind


def configure_logging():
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")


"""Reproducible, volume-based Pitzer chemistry for the article case."""


import sys
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parent


ROOT = CASE.parents[1]


from mf6pqc import MF6PQC

VM = dict(Halite=0.0271, Carnallite=0.1737, Sylvite=0.0375, Gypsum=0.0739, Polyhalite=0.218)


PHI = 0.20


FRACTIONS = dict(Halite=0.40, Carnallite=0.16, Sylvite=0.015, Gypsum=0.025, Polyhalite=0.04)


INERT = 1 - PHI - sum(FRACTIONS.values())


ELEMENTS = ["Na", "K", "Mg", "Ca", "Cl", "S(6)", "C(4)"]


def write_input():
    assert INERT > 0
    text = """TITLE Halite-saturated injection into a mixed evaporite assemblage
# Conditioning solutions are external reservoirs; their solids are not in the domain.
SOLUTION 90
    temp 25
    pH 6.818060
    units mol/L
    density 1.277848
    Cl 8.023463 charge
    K 0.444199
    Na 0.387030
    Ca 0.011049
    Mg 3.637958
    S(6) 0.051852
    C(4) 0.000088
EQUILIBRIUM_PHASES 90
    Halite 0 10
    Carnallite 0 10
    Sylvite 0 10
    Gypsum 0 10
    Polyhalite 0 10
SAVE solution 0
END
SOLUTION 91
    temp 25
    pH 7.314977
    units mol/L
    density 1.198844
    Cl 5.189443 charge
    K 0.000041
    Na 5.328907
    Ca 0.000031
    Mg 0.000515
    S(6) 0.070299
    C(4) 0.000088
EQUILIBRIUM_PHASES 91
    Halite 0 10
SAVE solution 1
END
# Finite in-domain inventory, mol per litre of representative BULK volume.
EQUILIBRIUM_PHASES 1
"""
    for name, fraction in FRACTIONS.items():
        text += f"    {name} 0 {fraction / VM[name]:.14g}\n"
    text += "END\nSELECTED_OUTPUT 1\n    -reset false\n    -high_precision true\nUSER_PUNCH 1\n"
    headings = (
        ELEMENTS
        + list(VM)
        + ["d_" + m for m in VM]
        + ["SI_" + m for m in VM]
        + ["water_kg", "solution_L", "pH", "RHO"]
    )
    text += "    -headings " + " ".join(headings) + "\n    -start\n"
    expressions = [f'TOT("{e}")*TOT("water")/SOLN_VOL' for e in ELEMENTS]
    expressions += [f'EQUI("{m}")' for m in VM]
    expressions += [f'EQUI_DELTA("{m}")' for m in VM]
    expressions += [f'SI("{m}")' for m in VM]
    expressions += ['TOT("water")', "SOLN_VOL", '-LA("H+")', "RHO"]
    for i, expr in enumerate(expressions, 1):
        text += f"    {i * 10} PUNCH {expr}\n"
    text += """    -end
END
KNOBS
    -iterations 400
    -step_size 10
    -diagonal_scale true
    -tolerance 1e-11
END
"""
    path = CASE / "input_data/input.pqi"
    path.write_text(text, encoding="utf-8")
    return path


def make_simulator(nxyz=1, threads=1, phi=PHI, scenario="S00", label="chemistry_check", **kwargs):
    return MF6PQC(
        case_name=label,
        nxyz=nxyz,
        nthreads=threads,
        temperature=25,
        pressure=1,
        porosity=phi,
        saturation=1,
        density=1.28,
        print_chemistry_mask=0,
        componentH2O=False,
        solution_density_volume=False,
        db_path=str(CASE / "input_data/pitzer.dat"),
        pqi_path=str(CASE / "input_data/input.pqi"),
        modflow_dll_path=library_path(),
        workspace=str(runtime_path(__file__, "simulation") / label),
        output_dir=str(runtime_path(__file__, "output") / label),
        if_update_density=scenario[1] == "1",
        use_phreeqc_calculated_density=True,
        if_update_porosity_K=scenario[2] == "1",
        mineral_molar_volumes=VM,
        fail_on_nonconvergence=True,
        **kwargs,
    )


"""Physical design. All four scenarios use the same grid, seed and forcing."""


from dataclasses import asdict, dataclass

SCENARIOS = {
    "S00": "Fixed density and structure",
    "S10": "Density only",
    "S01": "Structure only",
    "S11": "Density and structure",
}


@dataclass
class Config:
    nx: int = 200
    nz: int = 50
    length: float = 400.0
    height: float = 100.0
    width: float = 1.0
    days: float = 2400.0
    dt: float = 4.0
    save_every: float = 100.0
    kv_ratio: float = 0.3
    injection_rate: float = 5.0
    injection_depth: float = 10.0
    extraction_depth: float = 20.0
    screen_conductance: float = 100.0
    outlet_head: float = 110.0
    alpha_l: float = 2.0
    alpha_t: float = 0.2
    diffusion: float = 8.64e-5

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
        # Source rows run from the surface down. Geometric block averaging
        # preserves the supplied log-conductivity structure without smoothing
        # or changing its amplitude to encourage a particular outcome.
        source = np.load(CASE / "input_data/initial_hk_source.npy")
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


"""Confined section with shallow recharge and an outflow-only recovery screen."""


from pathlib import Path

import flopy

from mf6pqc.utils import get_gwt_model_name


def build_model(c, workspace, species, initial, injection, k, density, rho0):
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    sim = flopy.mf6.MFSimulation(
        sim_name="channel2d", sim_ws=str(workspace), exe_name=executable_path(), verbosity_level=0
    )
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(c.days, c.steps, 1.0)])
    dis = dict(
        nlay=c.nz,
        nrow=1,
        ncol=c.nx,
        delr=c.dx,
        delc=c.width,
        top=c.height,
        botm=c.height - np.arange(1, c.nz + 1) * c.dz,
    )
    flow = flopy.mf6.ModflowGwf(sim, modelname="gwf_model", save_flows=True)
    solver = flopy.mf6.ModflowIms(
        sim,
        filename="flow.ims",
        print_option="SUMMARY",
        outer_dvclose=1e-8,
        outer_maximum=150,
        inner_maximum=500,
        inner_dvclose=1e-9,
        rcloserecord=1e-7,
        linear_acceleration="BICGSTAB",
        scaling_method="NONE",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(solver, [flow.name])
    flopy.mf6.ModflowGwfdis(flow, pname="DIS", **dis)
    flopy.mf6.ModflowGwfic(flow, strt=c.outlet_head)
    flopy.mf6.ModflowGwfnpf(
        flow,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=k.reshape(c.shape),
        k33=(c.kv_ratio * k).reshape(c.shape),
    )
    # Quasi-steady pressure adjustment at every reaction/transport time step.
    # No compressibility, deformation, unsaturation or viscosity feedback.
    well = [
        ((int(layer), 0, 0), c.injection_rate / len(c.inlet_layers), *injection)
        for layer in c.inlet_layers
    ]
    flopy.mf6.ModflowGwfwel(
        flow,
        pname="INLET",
        filename="inlet.wel",
        save_flows=True,
        auxiliary=species,
        stress_period_data=well,
    )
    init = initial.reshape(len(species), c.nxyz)
    outlet = [
        (
            (int(layer), 0, c.nx - 1),
            c.outlet_head,
            c.screen_conductance / len(c.outlet_layers),
            *init[:, layer * c.nx + c.nx - 1],
        )
        for layer in c.outlet_layers
    ]
    flopy.mf6.ModflowGwfdrn(
        flow,
        pname="OUTLET",
        filename="outlet.drn",
        save_flows=True,
        auxiliary=species,
        stress_period_data=outlet,
    )
    records = [("HEAD", "STEPS", *c.save_steps), ("BUDGET", "STEPS", *c.save_steps)]
    flopy.mf6.ModflowGwfoc(
        flow,
        head_filerecord="flow.hds",
        budget_filerecord="flow.bud",
        saverecord=records,
        printrecord=[("BUDGET", "LAST")],
    )
    for i, e in enumerate(species):
        name = get_gwt_model_name(e)
        gwt = flopy.mf6.ModflowGwt(sim, modelname=name, save_flows=True)
        ims = flopy.mf6.ModflowIms(
            sim,
            filename=name + ".ims",
            print_option="SUMMARY",
            outer_dvclose=1e-8,
            outer_maximum=80,
            inner_dvclose=1e-9,
            inner_maximum=250,
            rcloserecord=1e-8,
            linear_acceleration="BICGSTAB",
            scaling_method="DIAGONAL",
            relaxation_factor=0.97,
        )
        sim.register_ims_package(ims, [name])
        flopy.mf6.ModflowGwtdis(gwt, pname="DIS", **dis)
        flopy.mf6.ModflowGwtic(gwt, strt=init[i].reshape(c.shape))
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=c.alpha_l,
            alv=c.alpha_l,
            ath1=c.alpha_t,
            ath2=c.alpha_t,
            atv=c.alpha_t,
            diffc=c.diffusion,
        )
        flopy.mf6.ModflowGwtmst(gwt, pname="MST", porosity=PHI)
        flopy.mf6.ModflowGwtssm(gwt, sources=[("INLET", "AUX", e), ("OUTLET", "AUX", e)])
        flopy.mf6.ModflowGwtoc(
            gwt, concentration_filerecord=name + ".ucn", saverecord=[("CONCENTRATION", "LAST")]
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea="gwf_model",
            exgmnameb=name,
            filename=name + ".gwfgwt",
        )
    if density:
        flopy.mf6.ModflowGwfbuy(
            flow,
            pname="BUY",
            denseref=rho0,
            nrhospecies=1,
            packagedata=[(0, 0.0, 0.0, get_gwt_model_name("Cl"), "CONCENTRATION")],
        )
    sim.write_simulation(silent=True)
    return sim
