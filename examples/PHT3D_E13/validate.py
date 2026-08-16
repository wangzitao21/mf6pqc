"""Basic numerical checks for the E13 breakthrough curves."""

from pathlib import Path

import numpy as np

from plot_utils import comparison_metrics, load_results


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
    for heading, values in comparison_metrics(CASE_DIR).items():
        print(
            f"{heading:>5}: NRMSE={values['nrmse']:.4f}, "
            f"correlation={values['correlation']:.4f}"
        )
        if values["nrmse"] > limits[heading]:
            raise AssertionError(f"{heading} NRMSE exceeds its regression limit")
        if values["correlation"] < 0.97:
            raise AssertionError(
                f"{heading} correlation is below its regression limit"
            )
    print("Validation passed.")


if __name__ == "__main__":
    main()
