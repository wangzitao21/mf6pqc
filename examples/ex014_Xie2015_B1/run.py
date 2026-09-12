"""Run the ex014_Xie2015_B1 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex014_Xie2015_B1.modflow_model import build_model
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
NLAY = 1
NROW = 1
NCOL = 81
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.35
DELR = [0.0125] + [0.025] * (NCOL - 2) + [0.0125]
DELC = [1.0]
TOP = 1.0
BOTM = 0
PERIOD_DATA = [(365.0 * 10, 10000, 1.0), (365.0 * 110, 110000, 1.0), (365.0 * 380, 38000, 1.0)]
ALH = 0.0
ATH1 = ALH / 10
DIFFC = 0.0
BOUNDARY_DISTANCE = DELR[0] / 2
INLET_HEAD = 0.007
OUTLET_HEAD = 0.0
VERTICAL_CONDUCTIVITY_RATIO = 0.1
HYDRAULIC_CONDUCTIVITY = 10.0
INITIAL_HEAD = 0.0


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    ic_mapping = {"solution": 0, "kinetics": 1}
    simulation_config = SimulationConfig(
        case_name="ex014",
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
            temperature_c=25.0,
            pressure_atm=2.0,
            porosity=POROSITY,
            saturation=1.0,
            density_kg_per_litre=1.0,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=False,
            use_solution_density_volume=False,
        ),
        feedback=FeedbackOptions(
            update_porosity_and_k=True,
            update_density=False,
            boundary_conductance_updates={
                "BUSHUI": {"cell_index": 0, "distance": BOUNDARY_DISTANCE},
                "GHB_RIGHT": {"cell_index": -1, "distance": BOUNDARY_DISTANCE},
            },
        ),
        output=OutputOptions(save_interval=1000, save_interval_offset=1),
    )
    hydraulic_conductivity = np.full((NLAY, NROW, NCOL), HYDRAULIC_CONDUCTIVITY)
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            period_data=PERIOD_DATA,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            vertical_conductivity_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            initial_head=INITIAL_HEAD,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            boundary_distance=BOUNDARY_DISTANCE,
        )
        simulator.run()
        simulator.save_results()
        print("ex014 done.")


if __name__ == "__main__":
    configure_logging()
    main()
