"""Run the MF6PQC reproduction of PHT3D Example 12."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from mf6pqc.mf6pqc import mf6pqc

from modflow_model import (
    NCOL,
    NXYZ,
    POROSITY,
    TRANSPORT_SUBSTEPS,
    coupling_period_data,
    coupling_step_end_times,
    transport_model,
)


def save_step_numbers(
    target_days: tuple[float, ...],
    coupling_times: np.ndarray,
) -> list[int]:
    """Map official PHT3D output times to one-based MF6 coupling steps."""
    saved_steps: list[int] = []
    for target in target_days:
        index = int(np.searchsorted(coupling_times, target))
        if index == len(coupling_times) or not np.isclose(
            coupling_times[index], target, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError(f"Official output time {target} is not a coupling step")
        saved_steps.append(index + 1)
    return [step * TRANSPORT_SUBSTEPS for step in saved_steps]


SAVE_HOURS = (1.5, 3.0, 12.5)
INPUT_DIR = CASE_DIR / "input_data"
# Use the actual PHT3D UCN output times selected by the verified notebook,
# rather than nominal hour labels or hand-copied save-time indices.
with np.load(INPUT_DIR / "official_reference.npz") as reference:
    SAVE_TARGET_DAYS = tuple(reference["actual_hours"] / 24.0)
    COUPLING_TIMES = coupling_step_end_times(reference["output_days"])
    PERIOD_DATA = coupling_period_data(reference["output_days"])
if len(PERIOD_DATA) != 4_388:
    raise ValueError(
        "Official E12 schedule must contain 4,388 reaction intervals; "
        f"reconstructed {len(PERIOD_DATA)}"
    )
SAVE_STEPS = save_step_numbers(SAVE_TARGET_DAYS, COUPLING_TIMES)
REACTION_STEPS = list(
    range(
        TRANSPORT_SUBSTEPS,
        len(PERIOD_DATA) * TRANSPORT_SUBSTEPS + 1,
        TRANSPORT_SUBSTEPS,
    )
)

params = {
    "case_name": "PHT3D_E12",
    "nxyz": NXYZ,
    "nthreads": 6,
    "temperature": 25.0,
    "pressure": 1.0,
    "porosity": POROSITY,
    "saturation": 1.0,
    "density": 1.0,
    "print_chemistry_mask": 0,
    # Transport water, excess H, and excess O separately.  PhreeqcRM documents
    # this as the robust formulation; transporting total H/O requires 8--10
    # accurate significant digits and visibly perturbs pH in this dilute case.
    "componentH2O": True,
    # PHT3D v2.10 transports pH/pe as legacy primary variables rather than
    # PhreeqcRM's signed charge imbalance.  Flooring Charge is an explicit,
    # case-local compatibility choice; the MF6PQC default remains signed.
    "signed_components": (),
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
    "save_steps": SAVE_STEPS,
    "reaction_steps": REACTION_STEPS,
    "progress_interval": 100 * TRANSPORT_SUBSTEPS,
    "fail_on_nonconvergence": True,
}

simulator = mf6pqc(**params)
initial_concentrations = simulator.setup(
    ic_map={"solution": 0, "surface": 1}
)
components = simulator.get_components()
pulse_concentrations = simulator.get_initial_concentrations(1)
chase_concentrations = simulator.get_initial_concentrations(0)

transport_model(
    sim_ws=params["workspace"],
    species_list=components,
    initial_conc=initial_concentrations,
    pulse_concentrations=pulse_concentrations,
    chase_concentrations=chase_concentrations,
    period_data=PERIOD_DATA,
    mf6_exe=REPOSITORY_DIR / "bin" / "mf6.7.0" / "mf6.exe",
)

try:
    simulator.run()
    simulator.save_results()
finally:
    simulator.finalize()

print(
    f"Saved target hours {SAVE_HOURS} at exact PHT3D coupling steps "
    f"{[step // TRANSPORT_SUBSTEPS for step in SAVE_STEPS]}; "
    f"{len(REACTION_STEPS)} reactions over "
    f"{len(PERIOD_DATA) * TRANSPORT_SUBSTEPS} MF6 transport steps."
)
