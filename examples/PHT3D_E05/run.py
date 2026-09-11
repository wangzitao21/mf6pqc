import sys

sys.dont_write_bytecode = True
import os
from pathlib import Path

from modflow_model import configure_logging, library_path, runtime_path, transport_model

from mf6pqc import MF6PQC


def main() -> None:
    example_dir = str(Path(__file__).resolve().parent)

    ic_mapping = {
        "solution": 0,
        "exchange": 1,
    }

    sim_params = {
        "case_name": "PHT3D_E05",
        "nxyz": 17,
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
        "pqi_path": os.path.join(example_dir, "input_data/phreeqc.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": runtime_path(__file__, "simulation"),
        "output_dir": runtime_path(__file__, "output"),
        "if_update_porosity_K": False,
        "if_update_density": False,
    }

    with MF6PQC(**sim_params) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        bc_conc = simulator.get_initial_concentrations(1)

        components = simulator.get_components()

        transport_model(
            sim_ws=runtime_path(__file__, "simulation"),
            species_list=components,
            initial_conc=initial_concentrations,
            bc=bc_conc,
        )

        simulator.run()
        # simulator.run_SIA()

        simulator.save_results()

        print("\n-------------------------------------------")
        print(f"{sim_params['case_name']} done")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    configure_logging()
    main()
