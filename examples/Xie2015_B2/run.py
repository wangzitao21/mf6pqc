import sys

sys.dont_write_bytecode = True
import os
from pathlib import Path

import numpy as np
from modflow_model import configure_logging, library_path, runtime_path, transport_model

from mf6pqc import MF6PQC

# todo 案例目录


def main() -> None:
    example_dir = str(Path(__file__).resolve().parent)

    ic_mapping = {
        "solution": 0,  # SOLUTION 0
        "kinetics": 1,  # 1
    }

    sim_params = {
        "case_name": "Xie2015_B2",
        # Match the B1-corrected MIN3P layout: two boundary half cells plus
        # 79 interior cells across the 2-m column.
        "nxyz": 81,
        "nthreads": 12,
        "temperature": 25.0,
        "pressure": 2.0,
        "porosity": 0.35,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        "solution_density_volume": False,
        "db_path": os.path.join(example_dir, "input_data/phreeqc.dat"),
        "pqi_path": os.path.join(example_dir, "input_data/phreeqc.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": runtime_path(__file__, "simulation"),
        "output_dir": runtime_path(__file__, "output"),
        "if_update_porosity_K": True,
        "if_update_density": False,
        # With 0.0008-year steps, save annual states so that the 10- and
        # 100-year benchmark profiles are stored exactly.
        "save_interval": 1250,
        "save_interval_offset": 1,
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

        # A standalone run of simulation/mf6.exe writes heads but does not execute
        # MF6PQC chemistry or produce these arrays. Verify that this Python driver
        # completed the reactive run before reporting success.
        output_dir = Path(sim_params["output_dir"])
        required_outputs = [
            output_dir / "results.npy",
            output_dir / "results_porosity.npy",
            output_dir / "results_K.npy",
        ]
        missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
        if missing_outputs:
            raise RuntimeError(
                "MF6PQC reactive outputs were not saved: " + ", ".join(missing_outputs)
            )
        print("MF6PQC reactive outputs saved successfully:")
        for path in required_outputs:
            print(f"  {path}  shape={np.load(path, mmap_mode='r').shape}")

        print("\n-------------------------------------------")
        print(f"'{sim_params['case_name']}' done")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    configure_logging()
    main()
