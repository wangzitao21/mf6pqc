import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

# todo 案例目录
example_dir = './examples/example3'

ic_mapping = {
    'solution': 0,   # 所有单元格使用 SOLUTION 0
    'exchange': 1,   # 所有单元格使用 EQUILIBRIUM_PHASES 1
}

sim_params = {
    "case_name": "PHT3D_CASE_5",
    "nxyz": 17,
    "nthreads": 6,

    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.35,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": os.path.join(example_dir, "input_data/phreeqc.dat"),
    "pqi_path": os.path.join(example_dir, "input_data/phreeqc.pqi"),
    "modflow_dll_path": "./bin/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),

    "if_update_porosity_K": False,
    "if_update_density": False
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
bc_conc = simulator.get_initial_concentrations(1)

components = simulator.get_components()

transport_model(
    sim_ws=os.path.join(example_dir, 'simulation'),
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