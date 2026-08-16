"""Native invariant check: chemistry must advance exactly once per logical step."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mf6pqc import MF6PQC
from mf6pqc.utils import get_gwt_model_name
from modflow_model import build_transport_model


METHODS = ("SNIA", "Strang", "SIA")
NCOL = 5
DAYS_PER_YEAR = 365.25
DECAY_RATE = 100.0 / DAYS_PER_YEAR
LOGICAL_DAMKOHLER = 0.4
LOGICAL_DT = LOGICAL_DAMKOHLER / DECAY_RATE


def run_method(method: str) -> np.ndarray:
    root = EXAMPLE_DIR / "simulation" / "reaction_only" / method.lower()
    output = EXAMPLE_DIR / "output" / "reaction_only" / method.lower()
    simulator = MF6PQC(
        case_name=f"reaction_only_{method.lower()}",
        nxyz=NCOL,
        nthreads=1,
        porosity=1.0,
        db_path=str(
            REPO_ROOT / "examples" / "PHT3D_E01" / "input_data" / "phreeqc.dat"
        ),
        pqi_path=str(EXAMPLE_DIR / "input_data" / "input.pqi"),
        modflow_dll_path=str(REPO_ROOT / "bin" / "mf6.7.0" / "libmf6.dll"),
        workspace=str(root),
        output_dir=str(output),
        progress_interval=100,
        sia_max_iterations=200,
        sia_atol=1.0e-12,
        sia_rtol=1.0e-10,
        sia_source_relaxation=0.7,
        sia_fail_on_nonconvergence=True,
    )
    try:
        initial = simulator.setup({"solution": 1, "kinetics": 1})
        boundary = simulator.get_initial_concentrations(1)
        spe_index = simulator.get_components().index("Spe")
        inlet_concentration = float(boundary[spe_index])
        build_transport_model(
            str(root),
            simulator.get_components(),
            initial,
            boundary,
            ncol=NCOL,
            length=1.0,
            porosity=1.0,
            # MODFLOW 6 rejects exactly zero K here.  This value makes the
            # transport change negligible over the single logical step.
            pore_velocity=1.0e-12,
            dispersivity=0.0,
            perlen=LOGICAL_DT,
            nstp=2 if method == "Strang" else 1,
        )
        getattr(simulator, f"run_{method}")()
        address = simulator.modflow_api.get_var_address(
            "X", get_gwt_model_name("Spe")
        )
        return np.asarray(
            simulator.modflow_api.get_value(address), dtype=float
        ).copy() / inlet_concentration
    finally:
        simulator.finalize()


def _persist(method: str, profile: np.ndarray) -> None:
    directory = EXAMPLE_DIR / "output" / "reaction_only" / method.lower()
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "profile.npy", profile)


def _launch(method: str) -> np.ndarray:
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--method", method],
        cwd=REPO_ROOT,
        check=True,
    )
    return np.load(
        EXAMPLE_DIR
        / "output"
        / "reaction_only"
        / method.lower()
        / "profile.npy"
    )


def run_check() -> None:
    profiles = {method: _launch(method) for method in METHODS}
    analytical = float(np.exp(-DECAY_RATE * LOGICAL_DT))
    pairwise_max = max(
        float(np.max(np.abs(profiles[first] - profiles[second])))
        for index, first in enumerate(METHODS)
        for second in METHODS[index + 1 :]
    )
    backend_error = max(
        float(np.max(np.abs(profile - analytical)))
        for profile in profiles.values()
    )
    report = {
        "logical_dt_days": LOGICAL_DT,
        "decay_rate_per_day": DECAY_RATE,
        "decay_rate_per_year": DECAY_RATE * DAYS_PER_YEAR,
        "damkohler_per_step": LOGICAL_DAMKOHLER,
        "analytical_concentration": analytical,
        "concentration_basis": "normalized C/C0",
        "method_means": {
            method: float(np.mean(profile)) for method, profile in profiles.items()
        },
        "maximum_pairwise_difference": pairwise_max,
        "maximum_error_against_continuous_analytic_decay": backend_error,
        "contract": (
            "All methods advance the saved PhreeqcRM kinetic state exactly once "
            "over one reaction-only logical interval."
        ),
    }
    output = EXAMPLE_DIR / "output" / "reaction_only"
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if pairwise_max > 1.0e-10:
        raise AssertionError(
            "SNIA, Strang, and SIA disagree in the reaction-only invariant"
        )
    if backend_error > 2.0e-3:
        raise AssertionError("Native kinetic integration drift exceeds 2e-3")
    print("Reaction-only kinetic-state invariant passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    arguments = parser.parse_args()
    os.chdir(REPO_ROOT)
    if arguments.method:
        _persist(arguments.method, run_method(arguments.method))
    else:
        run_check()
