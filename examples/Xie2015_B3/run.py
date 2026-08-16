import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

# todo 案例目录
example_dir = './examples/Xie2015_B3'

ic_mapping = {
    'solution':           0,   # SOLUTION 0
    'kinetics':           1,  # KINETICS 1
}

sim_params = {
    "case_name": "Xie2015_B3",
    # As in the verified B2 model, use 79 interior cells and two boundary
    # half cells.  This preserves the paper's 0.025-m spatial resolution
    # while placing the specified head and solution boundaries at x = 0 and 2 m.
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
    "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
    "modflow_dll_path": "./bin/mf6.7.0/libmf6.dll",
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),

    "if_update_porosity_K": True,
    "if_update_density": False,
    # The model uses 146,000 fixed 0.25-day steps over 100 years (initial
    # Courant number ~= 1). Save every 14,600 steps so the retained states
    # remain spaced by exactly 10 years for Fig. 4 and Fig. 5.
    "save_interval": 14600,
    "save_interval_offset": 1,
    # Print progress every 1,000 reaction steps. This affects console
    # feedback only; it does not alter the numerical solution.
    "progress_interval": 1000,
    # GHB conductances must evolve with the adjacent-cell conductivity.
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

output_dir = Path(sim_params["output_dir"])
required_outputs = [
    output_dir / "results.npy",
    output_dir / "results_porosity.npy",
    output_dir / "results_K.npy",
]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
if missing_outputs:
    raise RuntimeError("MF6PQC reactive outputs were not saved: " + ", ".join(missing_outputs))

simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")
