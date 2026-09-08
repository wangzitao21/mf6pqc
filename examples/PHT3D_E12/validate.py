"""Validate MF6PQC Example 12 against the official PHT3D profiles."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _example_support as _example_support
import numpy as np
from _example_support import runtime_path

CASE_DIR = Path(__file__).resolve().parent
REFERENCE_FILE = CASE_DIR / "input_data" / "official_reference.npz"
FOCUS_INTERVAL = (0.1, 0.175)


def profile_metrics(
    model: np.ndarray,
    reference: np.ndarray,
    focus: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return global NRMSE/correlation and focused RMSE/maximum error."""
    error = model - reference
    span = float(np.ptp(reference))
    nrmse = float(np.sqrt(np.mean(error**2)) / span)
    correlation = float(np.corrcoef(model, reference)[0, 1])
    focus_error = error[focus]
    focus_rmse = float(np.sqrt(np.mean(focus_error**2)))
    focus_max = float(np.max(np.abs(focus_error)))
    return nrmse, correlation, focus_rmse, focus_max


def main() -> None:
    results = np.load(runtime_path(__file__, "output") / "results.npy")
    result_times = np.load(runtime_path(__file__, "output") / "results_times.npy")
    headings = (
        (runtime_path(__file__, "output") / "results_headings.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    with np.load(REFERENCE_FILE) as source:
        official = {name: source[name].copy() for name in source.files}

    expected_shape = (4, len(headings), official["x"].size)
    if results.shape != expected_shape:
        raise AssertionError(f"Unexpected result shape {results.shape}; expected {expected_shape}")
    if not np.all(np.isfinite(results)):
        raise AssertionError("MF6PQC results contain non-finite values")
    np.testing.assert_allclose(
        result_times[1:],
        official["actual_hours"] / 24.0,
        rtol=0.0,
        atol=2.0e-10,
    )

    x = official["x"]
    focus = (x >= FOCUS_INTERVAL[0]) & (x <= FOCUS_INTERVAL[1])
    limits = {
        "U_6": {
            "nrmse": 0.018,
            "correlation": 0.999,
            "focus_rmse": 6.0e-7,
            "focus_max": 1.3e-6,
        },
        "pH": {
            "nrmse": 0.018,
            "correlation": 0.999,
            "focus_rmse": 0.010,
            "focus_max": 0.030,
        },
    }

    print("Official PHT3D comparison (focus: 0.1-0.175 m)")
    print(
        f"{'field':<6} {'hour':>9} {'NRMSE':>10} {'corr':>10} {'focus RMSE':>13} {'focus max':>12}"
    )
    for field, field_limits in limits.items():
        row = headings.index(field)
        for index, hour in enumerate(official["actual_hours"]):
            metrics = profile_metrics(results[index + 1, row], official[field][index], focus)
            nrmse, correlation, focus_rmse, focus_max = metrics
            print(
                f"{field:<6} {hour:9.5f} {nrmse:10.5f} {correlation:10.5f} "
                f"{focus_rmse:13.5g} {focus_max:12.5g}"
            )
            if nrmse > field_limits["nrmse"]:
                raise AssertionError(f"{field} NRMSE exceeds its regression limit")
            if correlation < field_limits["correlation"]:
                raise AssertionError(f"{field} correlation is below its regression limit")
            if focus_rmse > field_limits["focus_rmse"]:
                raise AssertionError(f"{field} focused RMSE exceeds its regression limit")
            if focus_max > field_limits["focus_max"]:
                raise AssertionError(f"{field} focused maximum error exceeds its regression limit")

    tracer_row = headings.index("Tracer")
    for index in range(2):
        tracer = results[index + 1, tracer_row]
        reference = official["Tracer"][index]
        tracer_nrmse = float(np.sqrt(np.mean((tracer - reference) ** 2)) / np.ptp(reference))
        if tracer_nrmse > 0.020:
            raise AssertionError("Tracer NRMSE exceeds its regression limit")
    final_tracer = float(np.max(np.abs(results[3, tracer_row])))
    if final_tracer > 7.0e-9:
        raise AssertionError("Final tracer residual exceeds its regression limit")

    print(f"Final tracer residual: {final_tracer:.5g} mol/L")
    print("PHT3D_E12 validation passed.")


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
