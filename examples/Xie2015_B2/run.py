import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

# todo 案例目录
example_dir = './examples/Xie2015_B2'

ic_mapping = {
    'solution':           0,   # SOLUTION 0
    'kinetics':           1,   # 1
}

sim_params = {
    "case_name": "Xie2015_B2",
    # Match the B1-corrected MIN3P layout: two boundary half cells plus
    # 79 interior cells across the 2-m column.
    "nxyz": 81,
    "nthreads": 12,
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
    "modflow_dll_path": "./bin/mf6.7.0/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),

    "if_update_porosity_K": True,
    "if_update_density": False,
    # With 0.0008-year steps, save annual states so that the 10- and
    # 100-year benchmark profiles are stored exactly.
    "save_interval": 1250,
    "save_interval_offset": 1,
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

# A standalone run of simulation/mf6.exe writes heads but does not execute
# MF6PQC chemistry or produce these arrays. Verify that this Python driver
# completed the reactive run before reporting success.
output_dir = Path(sim_params["output_dir"])
required_outputs = [
    output_dir / "results.npy",
    output_dir / "results_porosity.npy",
    output_dir / "results_K.npy",
]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
if missing_outputs:
    raise RuntimeError("MF6PQC reactive outputs were not saved: " + ", ".join(missing_outputs))
print("MF6PQC reactive outputs saved successfully:")
for path in required_outputs:
    print(f"  {path}  shape={np.load(path, mmap_mode='r').shape}")

simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")
