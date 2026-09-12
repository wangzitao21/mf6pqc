"""Run the ex004_PHT3D_04 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from ex004_PHT3D_04.modflow_model import build_model
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
NCOL = 40
NXYZ = NLAY * NROW * NCOL
DELR = 0.002
DELC = 1.0
TOP = 1.0
BOTM = 0.0
POROSITY = 1.0
SECONDS_PER_DAY = 86400.0
PERLEN = 20736.0 / SECONDS_PER_DAY
OUTPUT_INTERVALS = 120
NSTP = 2 * OUTPUT_INTERVALS
TIME_STEP = PERLEN / NSTP
INFLOW_RATE = 1.15741e-05 * SECONDS_PER_DAY
HYDRAULIC_CONDUCTIVITY = 1.0
ALH = 0.002
ATH1 = 0.0002
INITIAL_HEAD = 1.0
DIFFC = 0.0
OUTLET_HEAD = 1.0


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    simulation_config = SimulationConfig(
        case_name="ex004",
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
            pressure_atm=1.0,
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
        output=OutputOptions(save_steps=list(range(2, NSTP + 1, 2)), progress_interval=20),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "exchange": 1})
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
        if "Charge" in species:
            charge_index = species.index("Charge")
            initial_concentrations[charge_index * NXYZ : (charge_index + 1) * NXYZ] = 0.0
            inflow_concentrations[charge_index] = 0.0
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
            perlen=PERLEN,
            nstp=NSTP,
            porosity=POROSITY,
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            initial_head=INITIAL_HEAD,
            outlet_head=OUTLET_HEAD,
            inflow_rate=INFLOW_RATE,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        if simulator.results.shape[0] != OUTPUT_INTERVALS + 1:
            raise RuntimeError(
                f"Unexpected number of saved output intervals: {simulator.results.shape[0]}"
            )
        print("ex004 done.")


if __name__ == "__main__":
    configure_logging()
    main()
