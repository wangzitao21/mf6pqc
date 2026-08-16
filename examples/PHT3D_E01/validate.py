"""Quantitatively validate MF6PQC PHT3D Example 1."""

from __future__ import annotations

from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    results = np.load(CASE_DIR / "output" / "results.npy")
    headings = (CASE_DIR / "output" / "results_headings.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    times = np.load(CASE_DIR / "output" / "results_times.npy")
    reference = np.load(CASE_DIR / "input_data" / "official_reference.npz")["Spe"]

    expected_shape = (2, len(headings), reference.size)
    if results.shape != expected_shape:
        raise AssertionError(f"Unexpected result shape {results.shape}; expected {expected_shape}")
    if headings != ["Spe"]:
        raise AssertionError(f"Unexpected selected-output headings: {headings}")
    if not np.all(np.isfinite(results)):
        raise AssertionError("MF6PQC results contain non-finite values")
    np.testing.assert_allclose(times, [0.0, 1826.0], rtol=0.0, atol=1.0e-10)

    reproduced = results[-1, headings.index("Spe")]
    error = reproduced - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    value_range = float(np.ptp(reference))
    nrmse = rmse / value_range
    maximum_absolute_error = float(np.max(np.abs(error)))
    if nrmse > 0.02:
        raise AssertionError(f"NRMSE {nrmse:.6%} exceeds the 2% validation limit")
    if maximum_absolute_error > 0.05:
        raise AssertionError(
            f"Maximum absolute error {maximum_absolute_error:.6g} exceeds 0.05"
        )
    print(
        "PHT3D_E01 validation passed: "
        f"NRMSE={nrmse:.4%}, max_abs={maximum_absolute_error:.6g}"
    )


if __name__ == "__main__":
    main()

