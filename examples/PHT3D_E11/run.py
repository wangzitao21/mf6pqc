from __future__ import annotations

import os
import sys

import numpy as np


EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(EXAMPLE_DIR))
sys.path.insert(0, REPO_DIR)

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import NCOL, NLAY, NROW, NXYZ, transport_model


POROSITY = 0.30
PHT3D_REACTION_WATER_VOLUME_L = 1.0


# PHT3D uses two NAPL sources beginning at column 14 (one-based). The
# generated phinp.dat expands each explicit source cell with COPY KINETICS:
# the upper source spans columns 14-21 and layers 9-14, while the lower source
# spans columns 14-17 in layer 24.
kinetics_map = np.zeros((NLAY, NROW, NCOL), dtype=np.int32)
kinetics_map[8:14, 0, 13:21] = 1
kinetics_map[23, 0, 13:17] = 1

ic_mapping = {
    "solution": 0,
    "equilibrium_phases": 1,
    "kinetics": kinetics_map.ravel(),
}

input_data_dir = os.path.join(EXAMPLE_DIR, "input_data")
sim_params = {
    "case_name": "PHT3D_E11",
    "nxyz": NXYZ,
    "nthreads": 6,
    "temperature": 15.0,
    "pressure": 2.0,
    "porosity": POROSITY,
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
    # PHT3D's MMOC solver takes five reaction/transport substeps in each
    # 0.5-day flow step. Match that 0.1-day coupling interval while retaining
    # the official 5-day output sequence through day 60.
    "save_steps": list(range(50, 601, 50)),
    "progress_interval": 50,
    "fail_on_nonconvergence": True,
}

simulator = mf6pqc(**sim_params)

# PHT3D runs each grid-cell chemistry calculation with one litre of water.
# PhreeqcRM instead defaults to a one-litre representative (bulk) volume, so
# at porosity 0.30 its default reaction water volume would be only 0.30 L.
# Use water-based units and a 1 / porosity representative volume to retain the
# official PHT3D amounts and kinetic rates without calibrating rate constants.
simulator.phreeqc_rm.SetUnitsKinetics(1)
simulator.phreeqc_rm.SetUnitsPPassemblage(1)
simulator.phreeqc_rm.SetRepresentativeVolume(
    np.full(NXYZ, PHT3D_REACTION_WATER_VOLUME_L / POROSITY, dtype=float)
)

initial_concentrations = simulator.setup(ic_map=ic_mapping)
components = simulator.get_components()
ambient_concentrations = simulator.get_initial_concentrations(0)
recharge_concentrations = simulator.get_initial_concentrations(1)

# PHREEQC's transported Charge component is a numerical residual and can be
# slightly negative (about 4e-7 mol/L here). MODFLOW 6 correctly rejects a
# negative specified concentration, while PhreeqcRM reconstructs charge
# balance after every transport step. Use zero for that residual in MF6.
charge_index = components.index("Charge")
initial_concentrations[
    charge_index * NXYZ : (charge_index + 1) * NXYZ
] = 0.0
ambient_concentrations[charge_index] = 0.0
recharge_concentrations[charge_index] = 0.0

transport_model(
    sim_ws=sim_params["workspace"],
    species_list=components,
    initial_conc=initial_concentrations,
    ambient_concentrations=ambient_concentrations,
    recharge_concentrations=recharge_concentrations,
    mf6_exe=os.path.join(REPO_DIR, "bin", "mf6.7.0", "mf6.exe"),
    nstp=600,
)

try:
    simulator.run()
    simulator.save_results()
finally:
    simulator.finalize()

print("\n-------------------------------------------")
print(f"'{sim_params['case_name']}' done.")
print("-------------------------------------------\n")
