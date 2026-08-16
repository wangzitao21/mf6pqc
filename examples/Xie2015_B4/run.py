import os
import sys
from pathlib import Path
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

# todo 案例目录
example_dir = './examples/Xie2015_B4'

# B4 is sensitive to temporal operator splitting because every transport
# step is followed by an instantaneous equilibrium calculation. The
# environment overrides are intended for quick convergence checks; the
# defaults are the benchmark run.
TOTAL_YEARS = int(os.environ.get("MF6PQC_B4_TOTAL_YEARS", "3000"))
STEPS_PER_YEAR = int(os.environ.get("MF6PQC_B4_STEPS_PER_YEAR", "300"))
SAVE_EVERY_YEARS = int(os.environ.get("MF6PQC_B4_SAVE_EVERY_YEARS", "100"))
WORKSPACE = os.environ.get(
    "MF6PQC_B4_WORKSPACE", os.path.join(example_dir, "simulation")
)
OUTPUT_DIR = os.environ.get(
    "MF6PQC_B4_OUTPUT_DIR", os.path.join(example_dir, "output")
)
if TOTAL_YEARS % SAVE_EVERY_YEARS:
    raise ValueError("TOTAL_YEARS must be divisible by SAVE_EVERY_YEARS")
NSTP = TOTAL_YEARS * STEPS_PER_YEAR
SAVE_INTERVAL = SAVE_EVERY_YEARS * STEPS_PER_YEAR
SAVED_STATES = TOTAL_YEARS // SAVE_EVERY_YEARS

ic_mapping = {
    'solution':           0,   # SOLUTION 0
    'equilibrium_phases': 1,   # EQUILIBRIUM_PHASES 1
}

sim_params = {
    "case_name": "Xie2015_B4",
    # Xie et al. use a 2-m column with 79 full interior cells and two
    # boundary half cells at dx=0.025 m.
    "nxyz": 81,
    "nthreads": 12,
    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.35,
    "saturation": 1.0,
    "density": 1.0,
    # The two half cells are prescribed Dirichlet boundary states. Their
    # chemistry remains reactive, while porosity and transport properties
    # stay fixed at the benchmark values.
    "porosity_update_mask": np.r_[False, np.ones(79, dtype=bool), False],
    # Free-water diffusion coefficient: 1e-9 m2/s, converted to m2/day.
    "d0": 1.0e-9 * 86400.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,

    "db_path": os.path.join(example_dir, "input_data/phreeqc.dat"),
    "pqi_path": os.path.join(example_dir, "input_data/input.pqi"),
    "modflow_dll_path": "./bin/mf6.7.0/libmf6.dll",
    "workspace": WORKSPACE,
    "output_dir": OUTPUT_DIR,

    "if_update_porosity_K": True,
    "if_update_density": False,
    "if_update_diffc": True,
    # Retain one completed state every 100 years.
    "save_interval": SAVE_INTERVAL,
    "save_interval_offset": 1,
    "progress_interval": 1000,
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
    sim_ws=WORKSPACE,
    species_list=components,
    perlen=365.0 * TOTAL_YEARS,
    nstp=NSTP,
    initial_conc=initial_concentrations,
    bc=bc_conc,
    porosity=0.35,
    d0=sim_params["d0"],
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
    output_dir / "results_diffc.npy",
]
missing_outputs = [str(path) for path in required_outputs if not path.is_file()]
if missing_outputs:
    raise RuntimeError("MF6PQC reactive outputs were not saved: " + ", ".join(missing_outputs))

expected_frames = {
    "results.npy": SAVED_STATES + 1,          # includes initial chemistry
    "results_porosity.npy": SAVED_STATES + 1, # includes initial porosity
    "results_K.npy": SAVED_STATES + 1,        # includes initial K
    "results_diffc.npy": SAVED_STATES,        # completed snapshots only
}
for path in required_outputs:
    values = np.load(path, mmap_mode="r")
    if values.shape[0] != expected_frames[path.name]:
        raise RuntimeError(
            f"Expected {expected_frames[path.name]} frames in {path}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(f"Non-finite values found in {path}")

simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done")
print("-------------------------------------------\n")
