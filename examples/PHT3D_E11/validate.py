"""Validate the E11 reproduction against the checked-in PHT3D fields."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from modflow_model import BOTM, DELR, NCOL, NLAY, TOP


CARBON_STANDARD_RATIO = 0.0112372
SULFUR_STANDARD_RATIO = (1.021e-4 / 2.192e-3) / 1.010
ORGANIC_DISPLAY_THRESHOLD = 1.0e-8
PROFILE_X = 24.0

DISPLAY_SPECS = (
    ("Toluene", (0.0, 1.0e-3), 1.0e-3),
    ("delta13C Toluene", (-25.0, -18.0), None),
    ("Naphthalene", (0.0, 4.0e-4), 4.0e-4),
    ("delta13C Naphthalene", (-25.0, -18.0), None),
    ("Sulfate", (0.0, 3.0e-3), 3.0e-3),
    ("delta34S Sulfate", (10.0, 60.0), None),
)

# Concentration limits are percentages of the reference figure's colour
# range; isotope limits are per mil.  These leave a reproducibility margin
# above the verified values while still detecting the former 0.5--1.5
# per-mil heavy-isotope transport drift.
REGRESSION_LIMITS: dict[str, tuple[float, float]] = {
    "Toluene": (1.0, 1.0),
    "delta13C Toluene": (0.2, 0.2),
    "Naphthalene": (1.0, 1.0),
    "delta13C Naphthalene": (0.2, 0.2),
    "Sulfate": (1.2, 1.2),
    "delta34S Sulfate": (0.4, 0.4),
}


def isotope_delta(
    light: np.ndarray,
    heavy: np.ndarray,
    standard_ratio: float,
    threshold: float,
    background_delta: float,
) -> np.ndarray:
    """Return isotope delta while masking ratios below the display threshold."""
    total = light + heavy
    delta = np.full(total.shape, background_delta, dtype=float)
    valid = (
        np.isfinite(light)
        & np.isfinite(heavy)
        & (total > threshold)
        & (light > 0.0)
    )
    delta[valid] = (
        heavy[valid] / light[valid] / standard_ratio - 1.0
    ) * 1000.0
    delta[~np.isfinite(total)] = np.nan
    return delta


def derive_fields(source: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Derive the six quantities displayed in the published E11 figure."""
    toluene = source["Tolu_l"] + source["Tolu_h"]
    naphthalene = source["Naph_l"] + source["Naph_h"]
    sulfate = source["Sulf_l_6"] + source["Sulf_h_6"]
    return {
        "Toluene": toluene,
        "delta13C Toluene": isotope_delta(
            source["Tolu_l"],
            source["Tolu_h"],
            CARBON_STANDARD_RATIO,
            ORGANIC_DISPLAY_THRESHOLD,
            -25.0,
        ),
        "Naphthalene": naphthalene,
        "delta13C Naphthalene": isotope_delta(
            source["Naph_l"],
            source["Naph_h"],
            CARBON_STANDARD_RATIO,
            ORGANIC_DISPLAY_THRESHOLD,
            -25.0,
        ),
        "Sulfate": sulfate,
        "delta34S Sulfate": isotope_delta(
            source["Sulf_l_6"],
            source["Sulf_h_6"],
            SULFUR_STANDARD_RATIO,
            1.0e-12,
            10.0,
        ),
    }


def grid_centres() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return E11 x/z centres and their 2-D mesh."""
    x = np.cumsum(DELR) - 0.5 * DELR
    layer_tops = np.r_[TOP, BOTM[:-1]]
    z = 0.5 * (layer_tops + BOTM)
    x_grid, z_grid = np.meshgrid(x, z)
    return x, z, x_grid, z_grid


def load_case_fields() -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    """Load MF6PQC and official arrays and return their derived fields."""
    output_dir = CASE_DIR / "output"
    results = np.load(output_dir / "results.npy")
    result_times = np.load(output_dir / "results_times.npy")
    headings = [
        line.strip()
        for line in (output_dir / "results_headings.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    expected_shape = (result_times.size, len(headings), NLAY * NCOL)
    if results.shape != expected_shape:
        raise AssertionError(
            f"Unexpected result shape {results.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(results)):
        raise AssertionError("MF6PQC results contain non-finite values")

    reshaped = results.reshape(result_times.size, len(headings), NLAY, NCOL)
    model = {name: reshaped[:, index] for index, name in enumerate(headings)}
    with np.load(CASE_DIR / "input_data" / "official_reference.npz") as archive:
        reference = {name: archive[name].copy() for name in archive.files}
    reference_times = reference.pop("time_days")
    np.testing.assert_allclose(result_times, reference_times, rtol=0.0, atol=1.0e-10)
    if not np.isclose(result_times[-1], 60.0):
        raise AssertionError("The E11 benchmark requires the day-60 result")
    return result_times, model, derive_fields(model), derive_fields(reference)


def comparison_metrics() -> list[dict[str, float | int | str]]:
    """Calculate displayed-field and HR-MLW profile RMSE values."""
    _, model, model_fields, reference_fields = load_case_fields()
    x, z, x_grid, z_grid = grid_centres()
    well_column = int(np.argmin(np.abs(x - PROFILE_X)))
    wet = np.isfinite(reference_fields["Sulfate"][-1])
    figure_mask = (
        wet
        & (x_grid >= 4.0)
        & (x_grid <= 36.0)
        & (z_grid >= 32.5)
        & (z_grid <= 34.15)
    )
    profile_domain = (
        (z >= 32.5) & (z <= 34.15) & wet[:, well_column]
    )

    rows: list[dict[str, float | int | str]] = []
    for name, limits, concentration_scale in DISPLAY_SPECS:
        reproduced = np.clip(model_fields[name][-1], *limits)
        official = np.clip(reference_fields[name][-1], *limits)
        field_mask = figure_mask & np.isfinite(reproduced) & np.isfinite(official)
        profile_mask = profile_domain.copy()
        if name.startswith("delta13C"):
            concentration_name = name.removeprefix("delta13C ")
            organic_mask = (
                reference_fields[concentration_name][-1]
                > ORGANIC_DISPLAY_THRESHOLD
            )
            field_mask &= organic_mask
            profile_mask &= organic_mask[:, well_column]

        field_error = reproduced[field_mask] - official[field_mask]
        profile_error = (
            reproduced[:, well_column][profile_mask]
            - official[:, well_column][profile_mask]
        )
        field_rmse = float(np.sqrt(np.mean(field_error**2)))
        profile_rmse = float(np.sqrt(np.mean(profile_error**2)))
        unit = "per mil"
        if concentration_scale is not None:
            field_rmse = 100.0 * field_rmse / concentration_scale
            profile_rmse = 100.0 * profile_rmse / concentration_scale
            unit = "% colour range"
        rows.append(
            {
                "name": name,
                "field_rmse": field_rmse,
                "profile_rmse": profile_rmse,
                "unit": unit,
                "cells": int(field_mask.sum()),
            }
        )
    return rows


def main() -> None:
    rows = comparison_metrics()
    print("Official PHT3D day-60 comparison")
    print(
        f"{'quantity':<26} {'field RMSE':>12} {'HR-MLW RMSE':>14} "
        f"{'unit':>15} {'cells':>7}"
    )
    for row in rows:
        print(
            f"{row['name']:<26} {row['field_rmse']:12.4f} "
            f"{row['profile_rmse']:14.4f} {row['unit']:>15} "
            f"{row['cells']:7d}"
        )
        if row["name"] in REGRESSION_LIMITS:
            field_limit, profile_limit = REGRESSION_LIMITS[row["name"]]
            if row["field_rmse"] > field_limit:
                raise AssertionError(f"{row['name']} field RMSE exceeds its limit")
            if row["profile_rmse"] > profile_limit:
                raise AssertionError(f"{row['name']} HR-MLW RMSE exceeds its limit")
    print("PHT3D_E11 validation passed.")


if __name__ == "__main__":
    main()
