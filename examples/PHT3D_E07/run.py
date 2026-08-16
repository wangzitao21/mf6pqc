import sys
from pathlib import Path

import numpy as np

CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

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
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": str(CASE_DIR / "input_data" / "phreeqc.dat"),
    "pqi_path": str(CASE_DIR / "input_data" / "phreeqc.pqi"),
    "modflow_dll_path": str(REPOSITORY_DIR / "bin" / "mf6.7.0" / "libmf6.dll"),
    "workspace": str(CASE_DIR / "simulation"),
    "output_dir": str(CASE_DIR / "output"),

    "if_update_porosity_K": False,
    "if_update_density": False
}

simulator = mf6pqc(**sim_params)
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
    sim_ws=str(CASE_DIR / "simulation"),
    species_list=components,
    initial_conc=initial_concentrations,
    bc=bc_conc,
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"{sim_params['case_name']}' done")
print("-------------------------------------------\n")
