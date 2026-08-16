"""Run the MF6PQC reproduction of PHT3D Example 1."""

from __future__ import annotations

import os
import sys

import numpy as np


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(EXAMPLE_DIR))
sys.path.insert(0, REPO_DIR)

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import NXYZ, transport_model


# PHT3D uses ICBUND=-1 in the first cell: it is a fixed-concentration
# boundary, while the other 149 cells start with zero Species.
solution_map = np.zeros(NXYZ, dtype=np.int32)
solution_map[0] = 1
kinetics_map = np.ones(NXYZ, dtype=np.int32)
kinetics_map[0] = -1

ic_mapping = {
    "solution": solution_map,
    "kinetics": kinetics_map,
}

input_data_dir = os.path.join(EXAMPLE_DIR, "input_data")
sim_params = {
    "case_name": "PHT3D_E01",
    "nxyz": NXYZ,
    "nthreads": 6,
    "temperature": 25.0,
    "pressure": 2.0,
    "porosity": 0.25,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 0,
    "componentH2O": False,
    "solution_density_volume": False,
    "db_path": os.path.join(input_data_dir, "phreeqc.dat"),
    "pqi_path": os.path.join(input_data_dir, "input.pqi"),
    "modflow_dll_path": os.path.join(REPO_DIR, "bin", "mf6.7.0", "libmf6.dll"),
    "workspace": os.path.join(EXAMPLE_DIR, "simulation"),
    "output_dir": os.path.join(EXAMPLE_DIR, "output"),
    "if_update_porosity_K": False,
    "if_update_density": False,
    "save_steps": [200],
    "progress_interval": 20,
    "fail_on_nonconvergence": True,
}

simulator = mf6pqc(**sim_params)

try:
    initial_concentrations = simulator.setup(ic_map=ic_mapping)
    components = simulator.get_components()
    inflow_concentrations = simulator.get_initial_concentrations(1)

    # Charge is a PHREEQC numerical residual, not a transported chemical.
    # MODFLOW 6 rejects its small negative value as a specified concentration.
    charge_index = components.index("Charge")
    initial_concentrations[
        charge_index * NXYZ : (charge_index + 1) * NXYZ
    ] = 0.0
    inflow_concentrations[charge_index] = 0.0

    transport_model(
        sim_ws=sim_params["workspace"],
        species_list=components,
        initial_conc=initial_concentrations,
        inflow_concentrations=inflow_concentrations,
        mf6_exe=os.path.join(REPO_DIR, "bin", "mf6.7.0", "mf6.exe"),
    )

    # PHT3D uses sequential, non-iterative operator splitting.
    simulator.run()
    simulator.save_results()
finally:
    simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done.")
print("-------------------------------------------\n")
