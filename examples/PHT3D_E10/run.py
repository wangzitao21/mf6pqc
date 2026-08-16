import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model

nlay, nrow, ncol = 1, 40, 80

example_dir = os.path.dirname(os.path.abspath(__file__))
input_data_dir = os.path.join(example_dir, "input_data")
repo_dir = os.path.dirname(os.path.dirname(example_dir))
output_dir = os.path.join(example_dir, "output")
simulation_dir = os.path.join(example_dir, "simulation")
os.makedirs(output_dir, exist_ok=True)
for filename in (
    "results.npy",
    "results_headings.txt",
    "results_times.npy",
    "results_manifest.json",
):
    try:
        os.remove(os.path.join(output_dir, filename))
    except FileNotFoundError:
        pass

kinetics_mask = np.load(os.path.join(input_data_dir, "init_Benznapl.npy")).ravel()
kinetics_mask[kinetics_mask == 0.2] = 1
kinetics_mask = kinetics_mask.astype(int)

ic_mapping = {
    'solution': 0,
    'equilibrium_phases': 1,
    'kinetics': kinetics_mask,
}

hk = np.load(os.path.join(input_data_dir, "hk.npy")).reshape(nlay, nrow, ncol)
strt = np.load(os.path.join(input_data_dir, "strt.npy")).reshape(nlay, nrow, ncol)

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
    "modflow_dll_path": os.path.join(repo_dir, "bin", "mf6.7.0", "libmf6.dll"),
    "workspace": simulation_dir,
    "output_dir": output_dir,

    "if_update_porosity_K": False,
    "if_update_density": False,

    "save_interval": 1,
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)

# In the template, the left ICBUND=-1 cells retain the background starting
# concentrations. Therefore SOLUTION 0 initializes both the aquifer and the
# fixed-concentration inflow boundary.
background_concentrations = simulator.get_initial_concentrations(0)

components = simulator.get_components()

transport_model(
    sim_ws=simulation_dir,
    species_list=components,
    initial_conc=initial_concentrations,
    inflow_concentrations=background_concentrations,
    hk=hk,
    initial_head=strt,
    mf6_exe=os.path.join(repo_dir, "bin", "mf6.7.0", "mf6.exe"),
)

try:
    simulator.run()
    # Close native backends before replacing output files on Windows. Some
    # file-indexing/preview processes otherwise keep the previous result open.
    simulator.finalize()
    simulator.save_results()
finally:
    simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done.")
print("-------------------------------------------\n")
