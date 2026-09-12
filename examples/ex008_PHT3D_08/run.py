"""Run the ex008_PHT3D_08 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from ex008_PHT3D_08.modflow_model import build_model
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
NROW = 31
NCOL = 51
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.3
PERLEN = 1100
NSTP = 55
LENGTH_X = 510.0
LENGTH_Y = 310.0
TOP = 10.0
BOTM = 0.0
HYDRAULIC_CONDUCTIVITY = 50.0
ALH = 10.0
ATH1 = ALH * 0.3
ATV = ALH * 0.1
DIFFC = 0.0
INLET_HEAD = 100.0
OUTLET_HEAD = 99.0
INFLOW_RATE = 2.0
WELL_CELL = (0, 15, 15)


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    ic_mapping = {"solution": 0, "kinetics": 1}
    simulation_config = SimulationConfig(
        case_name="ex008",
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
        output=OutputOptions(progress_interval=10),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
        background_concentrations = simulator.get_initial_concentrations(0)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            background_concentrations=background_concentrations,
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
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            inflow_rate=INFLOW_RATE,
            well_cell=WELL_CELL,
            alh=ALH,
            ath1=ATH1,
            atv=ATV,
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        print("ex008 done.")


if __name__ == "__main__":
    configure_logging()
    main()
