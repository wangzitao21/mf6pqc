import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

example_dir = './examples/Xie2015_B1'

ic_mapping = {
    'solution': 0,   # SOLUTION 0
    'kinetics': 1,   # 1
}

sim_params = {
    "case_name": "Xie2015_B1",
    # Two 0.0125-m boundary half cells and 79 interior 0.025-m cells match
    # the MIN3P B1 reference geometry exactly.
    "nxyz": 81,
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
    "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
    "modflow_dll_path": "./bin/mf6.7.0/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),

    "if_update_porosity_K": True,
    "if_update_density": False,
    # With offset=1, each saved frame is an exact endpoint: annually through
    # 120 years and every 10 years thereafter, including 10, 100, 120 and 500.
    "save_interval": 1000,
    "save_interval_offset": 1,
    # Fixed-head faces lie 0.00625 m from the centers of the two boundary
    # half cells.  Their GHB conductance follows the adjacent K each step.
    "boundary_conductance_updates": {
        "BUSHUI": {"cell_index": 0, "distance": 0.00625},
        "GHB_RIGHT": {"cell_index": -1, "distance": 0.00625},
    },
}

K_arr = np.ones((1, 1, 81)) * 10.0

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
bc_conc = simulator.get_initial_concentrations(1)

components = simulator.get_components()

transport_model(
    nrow=1,
    ncol=81,
    nlay=1,
    sim_ws=os.path.join(example_dir, 'simulation'),
    species_list=components,
    initial_conc=initial_concentrations,
    bc=bc_conc,
    porosity=0.35,
    K11=K_arr,
    initial_head=0.0
)

simulator.run()
simulator.save_results()

simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")
