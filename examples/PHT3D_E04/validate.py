"""Quantitatively validate MF6PQC E04 against the official PHT3D output."""

from __future__ import annotations

from pathlib import Path

from plot_utils import comparison_metrics, load_comparison


CASE_DIR = Path(__file__).resolve().parent
MAX_NRMSE = 0.03
MIN_CORRELATION = 0.995


def main() -> None:
    _, mf6pqc, pht3d = load_comparison(CASE_DIR)
    metrics = comparison_metrics(mf6pqc, pht3d)

    print("species  NRMSE       correlation  RMSE (mmol/L)")
    for species, values in metrics.items():
        print(
            f"{species:>7}  {values['NRMSE']:.6f}   "
            f"{values['Correlation']:.6f}     "
            f"{values['RMSE (mmol/L)']:.6f}"
        )
        if values["NRMSE"] > MAX_NRMSE:
            raise AssertionError(f"{species} NRMSE exceeds {MAX_NRMSE}")
        if values["Correlation"] < MIN_CORRELATION:
            raise AssertionError(
                f"{species} correlation is below {MIN_CORRELATION}"
            )

    print("Validation passed.")


if __name__ == "__main__":
    main()
