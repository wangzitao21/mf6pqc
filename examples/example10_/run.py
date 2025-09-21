import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

# todo 案例目录
example_dir = './examples/example10_'

ic_mapping = {
    'solution':           0,   # SOLUTION 0
    'equilibrium_phases': 1,   # EQUILIBRIUM_PHASES 1
    'exchange':           1,   # EXCHANGE 1
}

sim_params = {
    "case_name": "Example10",
    "nxyz": 100*400,
    "nthreads": 12,
    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.30,
    "saturation": 1.0,
    "density": 1.277848,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": os.path.join(example_dir, "input_data/pitzer.dat"),
    "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
    "modflow_dll_path": "./bin/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output/B"),

    "if_update_porosity_K": True,
    "if_update_density": True
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
bc_conc = simulator.get_initial_concentrations(1)

components = simulator.get_components()

current_K_grid = np.load(os.path.join(example_dir, "./input_data/init_hk/hk_2_gaussian.npy")).reshape(100, 400, 1)

transport_model(
    sim_ws=os.path.join(example_dir, 'simulation'),
    species_list=components,
    initial_conc=initial_concentrations,
    bc=bc_conc,
    porosity=0.30,
    K11=current_K_grid,
    initial_density=1277.848,
    initial_head=100,
)

simulator.run()
simulator.save_results()

simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")