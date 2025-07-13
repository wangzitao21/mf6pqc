import os
import sys
import phreeqcrm
import numpy as np
import matplotlib.pyplot as plt
import time
import modflowapi

from mf6pqc.mf6pqc import mf6pqc

from modflow_models.modflow_model_10 import transport_model

kinetics_mask = np.load("./input_data/PHT3D_CASE_10/init_Benznapl.npy").ravel()
kinetics_mask[kinetics_mask == 0.2] = 1
kinetics_mask = kinetics_mask.astype(int)

ic_mapping = {
    'solution': 0,
    'equilibrium_phases': 1,
    'kinetics': kinetics_mask,
}

sim_params = {
    "case_name": "PHT3D_CASE_10",
    "nxyz": 3200,
    "nthreads": 6,

    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.30,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": "./input_data/PHT3D_CASE_10/phreeqc.dat", 
    "pqi_path": "./input_data/PHT3D_CASE_10/phreeqc.pqi",
    "modflow_dll_path": f"C:\\ProgramFiles\\MODFLOW\\libmf6.dll",
    "workspace": './simulation/PHT3D_CASE_10'
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)

bc_conc_1 = simulator.get_initial_concentrations(0)
bc_conc_15 = simulator.get_initial_concentrations(1)

components = simulator.get_components()

transport_model(
    sim_ws='./simulation/PHT3D_CASE_10',
    species_list=components,
    initial_conc=initial_concentrations,
    bc_1=bc_conc_1,
    bc_15=bc_conc_15
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"模拟 '{sim_params['case_name']}' 执行完毕。")
print("-------------------------------------------\n")