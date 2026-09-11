import sys

sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
from modflow_model import configure_logging, library_path, runtime_path, transport_model

from mf6pqc import MF6PQC


def main() -> None:
    CASE_DIR = Path(__file__).resolve().parent
    CASE_DIR.parents[1]

    nxyz = 41

    # PHT3D uses ICBUND=-1 for the first transport cell.  It is therefore a
    # fixed-concentration boundary cell and is not part of the 40 reactive cells.
    solution_ic = np.zeros(nxyz, dtype=int)
    solution_ic[0] = 1
    kinetics_ic = np.ones(nxyz, dtype=int)
    kinetics_ic[0] = -1

    ic_mapping = {
        "solution": solution_ic,
        "kinetics": kinetics_ic,
    }

    sim_params = {
        "case_name": "PHT3D_E07",
        "nxyz": nxyz,
        "nthreads": 6,
        "temperature": 25.0,
        "pressure": 2.0,
        "porosity": 1.0,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        "solution_density_volume": False,
        "db_path": str(CASE_DIR / "input_data" / "phreeqc.dat"),
        "pqi_path": str(CASE_DIR / "input_data" / "phreeqc.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": str(runtime_path(__file__, "simulation")),
        "output_dir": str(runtime_path(__file__, "output")),
        "if_update_porosity_K": False,
        "if_update_density": False,
    }

    with MF6PQC(**sim_params) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        bc_conc = simulator.get_initial_concentrations(1)

        components = simulator.get_components()
        s_a_index = components.index("S_a")

        # PHT3D applies the prescribed 1.0e-3 mol/L value directly to its
        # fixed-concentration transport cell.  Preserve that exact transport value
        # instead of the small PHREEQC molality/solution-volume conversion offset.
        initial_concentrations[s_a_index * nxyz] = 1.0e-3
        bc_conc[s_a_index] = 1.0e-3

        transport_model(
            sim_ws=str(runtime_path(__file__, "simulation")),
            species_list=components,
            initial_conc=initial_concentrations,
            bc=bc_conc,
        )

        simulator.run()
        simulator.save_results()

        print("\n-------------------------------------------")
        print(f"{sim_params['case_name']}' done")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    configure_logging()
    main()
