"""Basic numerical checks for the E13 breakthrough curves."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json

import _example_support as _example_support
import numpy as np
from _example_support import runtime_path
from plot import comparison_metrics, load_results

CASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    results, headings, _ = load_results(CASE_DIR)
    if results.shape != (165, 25, 16):
        raise AssertionError(f"Unexpected results shape: {results.shape}")
    for heading in ("Cl", "S_6", "Mg", "C_4", "pH", "Ca"):
        values = results[:, headings.index(heading), -1]
        if not np.all(np.isfinite(values)):
            raise AssertionError(f"{heading} contains non-finite values")
        print(f"{heading:>5}: {values.min():.6g} .. {values.max():.6g}")
    if results[:, headings.index("S_6"), -1].max() < 2.0e-3:
        raise AssertionError("The pyrite-oxidation sulfate peak was not reproduced")
    if results[:, headings.index("Cl"), -1].min() > 7.0e-3:
        raise AssertionError("The first-solution chloride breakthrough is absent")
    print("\nComparison with verified PHT3D notebook output:")
    limits = {
        "Cl": 0.060,
        "S_6": 0.065,
        "Mg": 0.040,
        "C_4": 0.055,
        "pH": 0.040,
        "Ca": 0.045,
    }
    metrics = comparison_metrics(CASE_DIR)
    failures = []
    for heading, values in metrics.items():
        print(f"{heading:>5}: NRMSE={values['nrmse']:.4f}, correlation={values['correlation']:.4f}")
        if values["nrmse"] > limits[heading]:
            failures.append(f"{heading} NRMSE exceeds {limits[heading]}")
        if values["correlation"] < 0.97:
            failures.append(f"{heading} correlation is below 0.97")
    report = {
        "metrics": metrics,
        "nrmse_limits": limits,
        "correlation_min": 0.97,
        "failures": failures,
        "passed": not failures,
    }
    output = runtime_path(__file__, "output")
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise AssertionError("; ".join(failures))
    print("Validation passed.")


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
