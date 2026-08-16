"""Run the MF6PQC reproduction of PHT3D Example 13."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import NCOL, NXYZ, POROSITY, transport_model


INPUT_DIR = CASE_DIR / "input_data"
zone_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
ic_mapping = {
    "solution": 0,
    "equilibrium_phases": zone_ids,
    "exchange": zone_ids,
    "surface": zone_ids,
    "kinetics": zone_ids,
}

params = {
    "case_name": "PHT3D_E13",
    "nxyz": NXYZ,
    "nthreads": 6,
    "temperature": 7.0,
    "pressure": 2.0,
    "porosity": POROSITY,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 0,
    "componentH2O": False,
    "solution_density_volume": False,
    "db_path": str(INPUT_DIR / "phreeqc.dat"),
    "pqi_path": str(INPUT_DIR / "input.pqi"),
    "modflow_dll_path": str(
        REPOSITORY_DIR / "bin" / "mf6.7.0" / "libmf6.dll"
    ),
    "workspace": str(CASE_DIR / "simulation"),
    "output_dir": str(CASE_DIR / "output"),
    "if_update_porosity_K": False,
    "if_update_density": False,
    "save_interval": 1,
    "progress_interval": 10,
    "fail_on_nonconvergence": True,
}

simulator = mf6pqc(**params)
initial_concentrations = simulator.setup(ic_map=ic_mapping)
components = simulator.get_components()
pulse_concentrations = simulator.get_initial_concentrations(1)
chase_concentrations = simulator.get_initial_concentrations(2)

transport_model(
    sim_ws=params["workspace"],
    species_list=components,
    initial_conc=initial_concentrations,
    pulse_concentrations=pulse_concentrations,
    chase_concentrations=chase_concentrations,
    mf6_exe=REPOSITORY_DIR / "bin" / "mf6.7.0" / "mf6.exe",
)

try:
    simulator.run()
    simulator.save_results()
finally:
    simulator.finalize()

print(f"Saved {NCOL}-cell outlet histories for both official stress periods.")
