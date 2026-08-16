"""Compare MF6PQC Example 8 with the official PHT3D 2.10 results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REFERENCE_FILE = CASE_DIR / "input_data" / "official_reference.npz"

NROW = 31
NCOL = 51
NTIME = 56  # initial state plus 55 transport steps
REFERENCE_SPECIES = ("Pce", "Vc", "Cl", "Na")


def load_results() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    headings = (CASE_DIR / "output" / "results_headings.txt").read_text().splitlines()
    raw = np.load(CASE_DIR / "output" / "results.npy")
    expected_shape = (NTIME, len(headings), NROW * NCOL)
    if raw.shape != expected_shape:
        raise ValueError(f"Expected result shape {expected_shape}, got {raw.shape}")

    mf6pqc = {
        name: raw[:, headings.index(name)].reshape(NTIME, NROW, NCOL)
        for name in REFERENCE_SPECIES
    }
    with np.load(REFERENCE_FILE) as reference:
        official = {name: reference[name] for name in REFERENCE_SPECIES}
    return mf6pqc, official


def main() -> None:
    mf6pqc, official = load_results()
    print("1100-day comparison against official PHT3D 2.10")
    print("species   RMSE (mol/L)   relative L2   correlation")

    correlations = {}
    relative_errors = {}
    for name in REFERENCE_SPECIES:
        actual = mf6pqc[name][-1]
        reference = official[name][-1]
        difference = actual - reference
        rmse = np.sqrt(np.mean(difference**2))
        relative_l2 = np.linalg.norm(difference) / np.linalg.norm(reference)
        correlation = np.corrcoef(actual.ravel(), reference.ravel())[0, 1]
        correlations[name] = correlation
        relative_errors[name] = relative_l2
        print(f"{name:>3s}      {rmse:11.4e}   {relative_l2:11.4e}   {correlation:11.6f}")

    if correlations["Pce"] < 0.995 or correlations["Vc"] < 0.995:
        raise AssertionError("PCE or VC plume geometry does not match official PHT3D")
    if relative_errors["Pce"] > 0.06 or relative_errors["Vc"] > 0.06:
        raise AssertionError("PCE or VC concentration error exceeds 6% in relative L2 norm")
    if relative_errors["Cl"] > 0.005 or relative_errors["Na"] > 0.001:
        raise AssertionError("Conservative/reactive tracer error exceeds tolerance")


if __name__ == "__main__":
    main()
