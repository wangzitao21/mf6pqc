"""Run the ex013_PHT3D_13 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex013_PHT3D_13.modflow_model import build_model
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
NCOL = 16
NXYZ = NLAY * NROW * NCOL
DELR = 0.0033125
DELC = 1.0
TOP = 0.00287433
BOTM = 0.0
POROSITY = 0.376
INFLOW_RATE = 0.00024
FLOW_PERIOD_DATA = [(0.9333333, 64, 1.0), (1.458333, 100, 1.0)]
TRANSPORT_SUBSTEPS = 4
PERIOD_DATA = [
    (length, steps * TRANSPORT_SUBSTEPS, multiplier)
    for length, steps, multiplier in FLOW_PERIOD_DATA
]
INITIAL_HEAD = 1.0
HYDRAULIC_CONDUCTIVITY = 1.0
ALH = 0.00537
ATH1 = 0.000537
DIFFC = 0.0
OUTLET_HEAD = 1.0


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    zone_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    ic_mapping = {
        "solution": 0,
        "equilibrium_phases": zone_ids,
        "exchange": zone_ids,
        "surface": zone_ids,
        "kinetics": zone_ids,
    }
    simulation_config = SimulationConfig(
        case_name="ex013",
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
            temperature_c=7.0,
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
        output=OutputOptions(save_interval=1, progress_interval=10),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        pulse_concentrations = simulator.get_initial_concentrations(1)
        chase_concentrations = simulator.get_initial_concentrations(2)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            pulse_concentrations=pulse_concentrations,
            chase_concentrations=chase_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            period_data=PERIOD_DATA,
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
        print(f"Saved {NCOL}-cell outlet histories for both stress periods.")


if __name__ == "__main__":
    configure_logging()
    main()
