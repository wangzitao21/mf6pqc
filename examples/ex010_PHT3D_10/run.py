"""Run the ex010_PHT3D_10 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex010_PHT3D_10.modflow_model import build_model
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
NROW = 40
NCOL = 80
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.3
PERLEN = 500
NSTP = 50
LENGTH_X = 200.0
LENGTH_Y = 50.0
TOP = 10.0
BOTM = 0.0
TSMULT = 1.0
ALH = 0.5
ATH1 = 0.1
DIFFC = 0.0
INLET_HEAD = 5.0
OUTLET_HEAD = 3.0


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    output_dir = runtime_path(__file__, "output")
    workspace = runtime_path(__file__, "simulation")
    kinetics_mask = np.load(INPUT_DIR / "benzene_napl.npy", allow_pickle=False).ravel()
    kinetics_mask[kinetics_mask == 0.2] = 1
    kinetics_mask = kinetics_mask.astype(int)
    ic_mapping = {"solution": 0, "equilibrium_phases": 1, "kinetics": kinetics_mask}
    hydraulic_conductivity = np.load(INPUT_DIR / "hk.npy", allow_pickle=False).reshape(
        NLAY, NROW, NCOL
    )
    initial_head = np.load(INPUT_DIR / "head.npy", allow_pickle=False).reshape(NLAY, NROW, NCOL)
    simulation_config = SimulationConfig(
        case_name="ex010",
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
        feedback=FeedbackOptions(update_porosity_and_k=False, update_density=False),
        output=OutputOptions(save_interval=1),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(0)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            length_x=LENGTH_X,
            length_y=LENGTH_Y,
            top=TOP,
            botm=BOTM,
            perlen=PERLEN,
            nstp=NSTP,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            initial_head=initial_head,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            tsmult=TSMULT,
        )
        simulator.run()
        simulator.save_results()
        print("ex010 done.")


if __name__ == "__main__":
    configure_logging()
    main()
