import os
import sys


example_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(os.path.dirname(example_dir))
sys.path.insert(0, repo_dir)

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import transport_model


ic_mapping = {
    "solution": 0,
    "kinetics": 1,
}

sim_params = {
    "case_name": "PHT3D_E09",
    "nxyz": 31 * 51,
    "nthreads": 6,
    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.30,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 1,
    "componentH2O": False,
    "solution_density_volume": False,
    "db_path": os.path.join(example_dir, "input_data", "phreeqc.dat"),
    "pqi_path": os.path.join(example_dir, "input_data", "input.pqi"),
    "modflow_dll_path": os.path.join(repo_dir, "bin", "mf6.7.0", "libmf6.dll"),
    "workspace": os.path.join(example_dir, "simulation"),
    "output_dir": os.path.join(example_dir, "output"),
    "if_update_porosity_K": False,
    "if_update_density": False,
    "save_interval": 1,
}

simulator = mf6pqc(**sim_params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
components = simulator.get_components()

transport_model(
    sim_ws=sim_params["workspace"],
    species_list=components,
    initial_conc=initial_concentrations,
    background_concentrations=simulator.get_initial_concentrations(0),
    well_concentrations=simulator.get_initial_concentrations(1),
    mf6_exe=os.path.join(repo_dir, "bin", "mf6.7.0", "mf6.exe"),
)

simulator.run()
simulator.save_results()
simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done.")
print("-------------------------------------------\n")
