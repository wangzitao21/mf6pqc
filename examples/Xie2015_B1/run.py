import sys

sys.dont_write_bytecode = True
import os
from pathlib import Path

import numpy as np
from modflow_model import configure_logging, library_path, runtime_path, transport_model

from mf6pqc import MF6PQC


def main() -> None:
    example_dir = str(Path(__file__).resolve().parent)

    ic_mapping = {
        "solution": 0,  # SOLUTION 0
        "kinetics": 1,  # 1
    }

    sim_params = {
        "case_name": "Xie2015_B1",
        # Two 0.0125-m boundary half cells and 79 interior 0.025-m cells match
        # the MIN3P B1 reference geometry exactly.
        "nxyz": 81,
        "nthreads": 6,
        "temperature": 25.0,
        "pressure": 2.0,
        "porosity": 0.35,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        "solution_density_volume": False,
        "db_path": os.path.join(example_dir, "input_data/phreeqc.dat"),
        "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": runtime_path(__file__, "simulation"),
        "output_dir": runtime_path(__file__, "output"),
        "if_update_porosity_K": True,
        "if_update_density": False,
        # With offset=1, each saved frame is an exact endpoint: annually through
        # 120 years and every 10 years thereafter, including 10, 100, 120 and 500.
        "save_interval": 1000,
        "save_interval_offset": 1,
        # Fixed-head faces lie 0.00625 m from the centers of the two boundary
        # half cells.  Their GHB conductance follows the adjacent K each step.
        "boundary_conductance_updates": {
            "BUSHUI": {"cell_index": 0, "distance": 0.00625},
            "GHB_RIGHT": {"cell_index": -1, "distance": 0.00625},
        },
    }

    K_arr = np.ones((1, 1, 81)) * 10.0

    with MF6PQC(**sim_params) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        bc_conc = simulator.get_initial_concentrations(1)

        components = simulator.get_components()

        transport_model(
            nrow=1,
            ncol=81,
            nlay=1,
            sim_ws=runtime_path(__file__, "simulation"),
            species_list=components,
            initial_conc=initial_concentrations,
            bc=bc_conc,
            porosity=0.35,
            K11=K_arr,
            initial_head=0.0,
        )

        simulator.run()
        simulator.save_results()

        print("\n-------------------------------------------")
        print(f"'{sim_params['case_name']}' done")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    configure_logging()
    main()
