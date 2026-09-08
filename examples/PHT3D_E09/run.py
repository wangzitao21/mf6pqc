import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os

import _example_support as _example_support
from _example_support import executable_path, library_path, runtime_path
from modflow_model import transport_model

from mf6pqc import MF6PQC


def main() -> None:
    example_dir = os.path.dirname(os.path.abspath(__file__))

    ic_mapping = {
        "solution": 0,
        "kinetics": 1,
    }

    sim_params = {
        "case_name": "PHT3D_E09",
        "nxyz": 31 * 51,
        "nthreads": 6,
        "temperature": 25.0,
        "pressure": 2.0,
        "porosity": 0.30,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        "solution_density_volume": False,
        "db_path": os.path.join(example_dir, "input_data", "phreeqc.dat"),
        "pqi_path": os.path.join(example_dir, "input_data", "input.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": runtime_path(__file__, "simulation"),
        "output_dir": runtime_path(__file__, "output"),
        "if_update_porosity_K": False,
        "if_update_density": False,
        "save_interval": 1,
    }

    with MF6PQC(**sim_params) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        components = simulator.get_components()

        transport_model(
            sim_ws=sim_params["workspace"],
            species_list=components,
            initial_conc=initial_concentrations,
            background_concentrations=simulator.get_initial_concentrations(0),
            well_concentrations=simulator.get_initial_concentrations(1),
            mf6_exe=executable_path("mf6.8.0"),
        )

        simulator.run()
        simulator.save_results()

        print("\n-------------------------------------------")
        print(f"'{sim_params['case_name']}' done.")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
