"""Run the ex017_Xie2015_B4 reactive-transport benchmark."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex017_Xie2015_B4.modflow_model import build_model
from example_utils import configure_logging, library_path, require_output_files, runtime_path

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
ALH = 0.0
ATH1 = 0.0
BOUNDARY_HEAD = 0.0
VERTICAL_CONDUCTIVITY_RATIO = 0.1
HYDRAULIC_CONDUCTIVITY = 10.0
INITIAL_HEAD = 0.0


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    total_years = int(os.environ.get("MF6PQC_B4_TOTAL_YEARS", "3000"))
    steps_per_year = int(os.environ.get("MF6PQC_B4_STEPS_PER_YEAR", "300"))
    save_every_years = int(os.environ.get("MF6PQC_B4_SAVE_EVERY_YEARS", "100"))
    workspace = runtime_path(__file__, "simulation", override=os.environ.get("MF6PQC_B4_WORKSPACE"))
    output_dir = runtime_path(__file__, "output", override=os.environ.get("MF6PQC_B4_OUTPUT_DIR"))
    if min(total_years, steps_per_year, save_every_years) <= 0:
        raise ValueError("Duration, steps per year, and save interval must be positive")
    if total_years % save_every_years:
        raise ValueError("TOTAL_YEARS must be divisible by SAVE_EVERY_YEARS")
    nstp = total_years * steps_per_year
    save_interval = save_every_years * steps_per_year
    saved_states = total_years // save_every_years
    ic_mapping = {"solution": 0, "equilibrium_phases": 1}
    simulation_config = SimulationConfig(
        case_name="ex017",
        nxyz=NXYZ,
        nthreads=12,
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
            free_water_diffusion_model_units=1e-09 * 86400.0,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=False,
            use_solution_density_volume=False,
        ),
        feedback=FeedbackOptions(
            update_porosity_and_k=True,
            update_density=False,
            update_diffusion=True,
            porosity_update_mask=np.r_[False, np.ones(NCOL - 2, dtype=bool), False],
        ),
        output=OutputOptions(
            save_interval=save_interval, save_interval_offset=1, progress_interval=1000
        ),
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
            perlen=365.0 * total_years,
            nstp=nstp,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            vertical_conductivity_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            initial_head=INITIAL_HEAD,
            alh=ALH,
            ath1=ATH1,
            d0=simulation_config.fields.free_water_diffusion_model_units,
            boundary_head=BOUNDARY_HEAD,
        )
        simulator.run()
        simulator.save_results()
        required_outputs = require_output_files(
            output_dir,
            ("results.npy", "results_porosity.npy", "results_K.npy", "results_diffc.npy"),
        )
        expected_frames = {
            "results.npy": saved_states + 1,
            "results_porosity.npy": saved_states + 1,
            "results_K.npy": saved_states + 1,
            "results_diffc.npy": saved_states,
        }
        for path in required_outputs:
            values = np.load(path, mmap_mode="r", allow_pickle=False)
            if values.shape[0] != expected_frames[path.name]:
                raise RuntimeError(
                    f"Expected {expected_frames[path.name]} frames in {path}, got {values.shape}"
                )
            if not np.isfinite(values).all():
                raise RuntimeError(f"Non-finite values found in {path}")
        print("ex017 done.")


if __name__ == "__main__":
    configure_logging()
    main()
