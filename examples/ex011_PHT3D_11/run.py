"""Run the ex011_PHT3D_11 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex011_PHT3D_11.modflow_model import build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    FeedbackOptions,
    OutputOptions,
    SimulationConfig,
)

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
POROSITY = 0.3
NLAY = 58
NROW = 1
NCOL = 99
NXYZ = NLAY * NROW * NCOL
FLOW_STEPS = 120
TRANSPORT_SUBSTEPS = 5
TOTAL_TRANSPORT_STEPS = FLOW_STEPS * TRANSPORT_SUBSTEPS
DELR = np.array([1.0] * 7 + [0.5] * 2 + [0.25] * 76 + [0.5] * 2 + [1.0] * 12, dtype=float)
DELC = np.array([1.0], dtype=float)
TOP = 34.2
BOTM = np.array(
    [34.15 - 0.05 * layer for layer in range(44)]
    + [31.75, 31.5, 31.25, 31.0, 30.5, 30.0]
    + list(np.arange(29.0, 21.0, -1.0)),
    dtype=float,
)
PERLEN = 60.0
LEFT_RATES = np.r_[
    np.full(36, 0.0267608), np.full(4, 0.1338042), np.full(2, 0.2676083), np.full(8, 0.5352167)
]
HYDRAULIC_CONDUCTIVITY = 86.4
ALH = 0.05
ALV = 0.05
ATH1 = 0.005
ATH2 = 0.0005
ATV = 0.005
DIFFC = 0.0
OUTLET_HEAD = 33.76
BOUNDARY_CONDUCTIVITY = 1000.0
RECHARGE_RATE = 0.001
FIRST_BOUNDARY_LAYER = 8


def cell_centers_x() -> np.ndarray:
    return np.cumsum(DELR) - 0.5 * DELR


def cell_centers_z() -> np.ndarray:
    layer_tops = np.r_[TOP, BOTM[:-1]]
    return 0.5 * (layer_tops + BOTM)


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    reaction_water_volume_l = 1.0
    reaction_steps = list(range(TRANSPORT_SUBSTEPS, TOTAL_TRANSPORT_STEPS + 1, TRANSPORT_SUBSTEPS))
    save_steps = list(range(50, TOTAL_TRANSPORT_STEPS + 1, 50))
    kinetics_map = np.zeros((NLAY, NROW, NCOL), dtype=np.int32)
    kinetics_map[8:14, 0, 13:21] = 1
    kinetics_map[23, 0, 13:17] = 1
    ic_mapping = {"solution": 0, "equilibrium_phases": 1, "kinetics": kinetics_map.ravel()}
    simulation_config = SimulationConfig(
        case_name="ex011",
        nxyz=NXYZ,
        nthreads=6,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=15.0,
            pressure_atm=2.0,
            porosity=POROSITY,
            saturation=1.0,
            density_kg_per_litre=1.0,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=False,
            use_solution_density_volume=False,
            signed_components=(),
        ),
        feedback=FeedbackOptions(update_porosity_and_k=False, update_density=False),
        output=OutputOptions(save_steps=save_steps, progress_interval=50),
        reaction_steps=reaction_steps,
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        simulator.phreeqc_rm.SetUnitsKinetics(1)
        simulator.phreeqc_rm.SetUnitsPPassemblage(1)
        simulator.phreeqc_rm.SetRepresentativeVolume(
            np.full(NXYZ, reaction_water_volume_l / POROSITY, dtype=float)
        )
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        ambient_concentrations = simulator.get_initial_concentrations(0)
        recharge_concentrations = simulator.get_initial_concentrations(1)
        charge_index = species.index("Charge")
        initial_concentrations[charge_index * NXYZ : (charge_index + 1) * NXYZ] = 0.0
        ambient_concentrations[charge_index] = 0.0
        recharge_concentrations[charge_index] = 0.0
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            ambient_concentrations=ambient_concentrations,
            recharge_concentrations=recharge_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            perlen=PERLEN,
            nstp=TOTAL_TRANSPORT_STEPS,
            porosity=POROSITY,
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            outlet_head=OUTLET_HEAD,
            recharge_rate=RECHARGE_RATE,
            alh=ALH,
            alv=ALV,
            ath1=ATH1,
            ath2=ATH2,
            atv=ATV,
            diffc=DIFFC,
            boundary_conductivity=BOUNDARY_CONDUCTIVITY,
            first_boundary_layer=FIRST_BOUNDARY_LAYER,
            left_rates=LEFT_RATES,
        )
        simulator.run()
        simulator.save_results()
        print("ex011 done.")


if __name__ == "__main__":
    configure_logging()
    main()
