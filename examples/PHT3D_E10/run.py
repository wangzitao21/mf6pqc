import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

nlay, nrow, ncol = 1, 40, 80

example_dir = './examples/PHT3D_E10'

kinetics_mask = np.load("./examples/PHT3D_E10/input_data/init_Benznapl.npy").ravel()
kinetics_mask[kinetics_mask == 0.2] = 1
kinetics_mask = kinetics_mask.astype(int)

ic_mapping = {
    'solution': 0,
    'equilibrium_phases': 1,
    'kinetics': kinetics_mask,
}

hk = np.load("./examples/PHT3D_E10/input_data/hk.npy").reshape(nlay, nrow, ncol)

sim_params = {
    "case_name": "PHT3D_E10",
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

    "db_path": os.path.join(example_dir, "input_data/phreeqc.dat"),
    "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
    "modflow_dll_path": "./bin/mf6.7.0/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),

    "if_update_porosity_K": False,
    "if_update_density": False,

    "save_interval": 1,
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)

bc_conc_1 = simulator.get_initial_concentrations(0)
bc_conc_15 = simulator.get_initial_concentrations(1)

components = simulator.get_components()

transport_model(
    sim_ws=os.path.join(example_dir, 'simulation'),
    species_list=components,
    initial_conc=initial_concentrations,
    bc_1=bc_conc_1,
    bc_15=bc_conc_15,
    hk=hk
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done.")
print("-------------------------------------------\n")