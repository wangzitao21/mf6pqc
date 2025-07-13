import os
import sys
import phreeqcrm
import numpy as np
import matplotlib.pyplot as plt
import time
import modflowapi

from scr.mf6pqc import mf6pqc

from modflow_models.modflow_model_7 import create_and_run_models

ic_mapping = {
    'solution': 0,
    'kinetics': 1,
}

sim_params = {
    "case_name": "PHT3D_CASE_7",
    "nxyz": 41,
    "nthreads": 6,

    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 1.0,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": "./input_data/PHT3D_CASE_7/phreeqc.dat", 
    "pqi_path": "./input_data/PHT3D_CASE_7/phreeqc.pqi",
    "modflow_dll_path": f"C:\\ProgramFiles\\MODFLOW\\libmf6.dll",
    "workspace": './simulation/PHT3D_CASE_7'
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
bc_conc = simulator.get_initial_concentrations(1)

components = simulator.get_components()

create_and_run_models(
    sim_ws='./simulation/PHT3D_CASE_7',
    species_list=components,
    initial_conc=initial_concentrations,
    bc=bc_conc,
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"模拟 '{sim_params['case_name']}' 执行完毕。")
print("-------------------------------------------\n")