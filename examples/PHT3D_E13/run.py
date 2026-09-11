"""Run the MF6PQC reproduction of PHT3D Example 13."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
from modflow_model import (
    NCOL,
    NXYZ,
    POROSITY,
    configure_logging,
    executable_path,
    library_path,
    runtime_path,
    transport_model,
)

from mf6pqc import MF6PQC


def main() -> None:
    CASE_DIR = Path(__file__).resolve().parent

    INPUT_DIR = CASE_DIR / "input_data"
    zone_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    ic_mapping = {
        "solution": 0,
        "equilibrium_phases": zone_ids,
        "exchange": zone_ids,
        "surface": zone_ids,
        "kinetics": zone_ids,
    }

    params = {
        "case_name": "PHT3D_E13",
        "nxyz": NXYZ,
        "nthreads": 6,
        "temperature": 7.0,
        "pressure": 1.0,
        "porosity": POROSITY,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        "solution_density_volume": False,
        "db_path": str(INPUT_DIR / "phreeqc.dat"),
        "pqi_path": str(INPUT_DIR / "input.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": str(runtime_path(__file__, "simulation")),
        "output_dir": str(runtime_path(__file__, "output")),
        "if_update_porosity_K": False,
        "if_update_density": False,
        "save_interval": 1,
        "progress_interval": 10,
        "fail_on_nonconvergence": True,
    }

    with MF6PQC(**params) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        components = simulator.get_components()
        pulse_concentrations = simulator.get_initial_concentrations(1)
        chase_concentrations = simulator.get_initial_concentrations(2)

        transport_model(
            sim_ws=params["workspace"],
            species_list=components,
            initial_conc=initial_concentrations,
            pulse_concentrations=pulse_concentrations,
            chase_concentrations=chase_concentrations,
            mf6_exe=executable_path(),
        )

        simulator.run()
        simulator.save_results()

        print(f"Saved {NCOL}-cell outlet histories for both stress periods.")


if __name__ == "__main__":
    configure_logging()
    main()
