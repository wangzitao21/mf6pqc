"""Run the MF6PQC reproduction of PHT3D Example 4."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from mf6pqc.mf6pqc import mf6pqc  # noqa: E402

from modflow_model import (  # noqa: E402
    NXYZ,
    OUTPUT_INTERVALS,
    POROSITY,
    TIME_STEPS,
    transport_model,
)


simulator = mf6pqc(
    case_name="PHT3D_E04",
    nxyz=NXYZ,
    nthreads=6,
    temperature=25.0,
    pressure=1.0,
    porosity=POROSITY,
    saturation=1.0,
    density=1.0,
    print_chemistry_mask=0,
    componentH2O=False,
    solution_density_volume=False,
    db_path=str(CASE_DIR / "input_data" / "phreeqc.dat"),
    pqi_path=str(CASE_DIR / "input_data" / "input.pqi"),
    modflow_dll_path=str(REPOSITORY_DIR / "bin" / "mf6.7.0" / "libmf6.dll"),
    workspace=str(CASE_DIR / "simulation"),
    output_dir=str(CASE_DIR / "output"),
    if_update_porosity_K=False,
    if_update_density=False,
    save_steps=list(range(2, TIME_STEPS + 1, 2)),
    progress_interval=20,
    fail_on_nonconvergence=True,
)

try:
    initial_concentrations = simulator.setup(
        ic_map={"solution": 0, "exchange": 1}
    )
    components = simulator.get_components()
    inflow_concentrations = simulator.get_initial_concentrations(1)

    # Charge is a numerical residual rather than an independently transported
    # chemical component. Avoid tiny negative specified concentrations.
    if "Charge" in components:
        charge_index = components.index("Charge")
        initial_concentrations[
            charge_index * NXYZ : (charge_index + 1) * NXYZ
        ] = 0.0
        inflow_concentrations[charge_index] = 0.0

    transport_model(
        sim_ws=CASE_DIR / "simulation",
        species_list=components,
        initial_conc=initial_concentrations,
        inflow_concentrations=inflow_concentrations,
        mf6_exe=REPOSITORY_DIR / "bin" / "mf6.7.0" / "mf6.exe",
    )
    simulator.run()
    simulator.save_results()
    if simulator.results.shape[0] != OUTPUT_INTERVALS + 1:
        raise RuntimeError(
            "Unexpected number of saved output intervals: "
            f"{simulator.results.shape[0]}"
        )
finally:
    simulator.finalize()

print("\n-------------------------------------------")
print("PHT3D_E04 done")
print("-------------------------------------------\n")
