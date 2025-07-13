import os
import sys
import numpy as np
import matplotlib.pyplot as plt

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from mf6pqc.mf6pqc import mf6pqc

from modflow_model import create_and_run_models

ic_mapping = {
    'solution':           0,   # SOLUTION 0
    'equilibrium_phases': 1,   # EQUILIBRIUM_PHASES 1
}

sim_params = {
    "case_name": "Example1",
    "nxyz": 50,
    "nthreads": 6,
    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.32,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": "./examples/example1/input_data/phreeqc.dat", 
    "pqi_path": "./examples/example1/input_data/phreeqc.pqi",
    "modflow_dll_path": "./bin/libmf6.dll",
    "workspace": './examples/example1/simulation',
    "output_dir": "./examples/example1/output",

    "if_update_porosity_K": False
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
bc_conc = simulator.get_initial_concentrations(1)

components = simulator.get_components()

create_and_run_models(
    sim_ws='./examples/example1/simulation',
    species_list=components,
    initial_conc=initial_concentrations,
    bc=bc_conc,
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")