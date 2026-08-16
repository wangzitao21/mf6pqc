"""Plotting helpers for PHT3D Example 13."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FLOW_RATE = 2.4e-4
PERIOD_DATA = ((0.9333333, 64, 1.0), (1.458333, 100, 1.0))
def output_times() -> np.ndarray:
    values = [0.0]
    elapsed = 0.0
    for period_length, step_count, _ in PERIOD_DATA:
        dt = period_length / step_count
        for _ in range(step_count):
            elapsed += dt
            values.append(elapsed)
    return np.asarray(values)


def load_results(case_dir: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    results = np.load(case_dir / "output" / "results.npy")
    headings = (
        case_dir / "output" / "results_headings.txt"
    ).read_text().splitlines()
    observations = np.loadtxt(case_dir / "input_data" / "appelo98_data.txt")
    return results, headings, observations


def load_official_reference(case_dir: Path) -> dict[str, np.ndarray]:
    """Load outlet histories extracted from the verified PHT3D notebook."""
    return dict(np.load(case_dir / "input_data" / "official_reference.npz"))


def comparison_metrics(case_dir: Path) -> dict[str, dict[str, float]]:
    """Compare MF6PQC outlet histories with PHT3D at identical times."""
    results, headings, _ = load_results(case_dir)
    reference = load_official_reference(case_dir)
    model_times = output_times()
    metrics = {}
    for heading in ("Cl", "S_6", "Mg", "C_4", "pH", "Ca"):
        model = np.interp(
            reference["time_days"],
            model_times,
            results[:, headings.index(heading), -1],
        )
        target = reference[heading]
        rmse = float(np.sqrt(np.mean((model - target) ** 2)))
        metrics[heading] = {
            "rmse": rmse,
            "nrmse": rmse / float(np.ptp(target)),
            "correlation": float(np.corrcoef(model, target)[0, 1]),
        }
    return metrics


def plot_breakthrough(case_dir: Path) -> tuple[plt.Figure, np.ndarray]:
    results, headings, observations = load_results(case_dir)
    reference = load_official_reference(case_dir)
    volume_ml = output_times() * FLOW_RATE * 1.0e6
    reference_volume_ml = reference["time_days"] * FLOW_RATE * 1.0e6
    outlet = results[:, :, -1]
    panels = (
        ("Cl", "Cl", 4, "blue", (0, 0.025)),
        (r"SO$_4^{2-}$", "S_6", 5, "blue", (0, 0.010)),
        ("Mg", "Mg", 1, "red", (0, 0.025)),
        # The verified PHT3D notebook plots PHT3D001.UCN here, which is
        # transported component C(4), despite the bicarbonate panel label.
        (r"HCO$_3^-$", "C_4", 3, "red", (0, 0.015)),
        ("pH", "pH", 6, "green", (4, 10)),
        (r"Ca$^{2+}$", "Ca", 2, "green", (0, 0.006)),
    )

    fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.0), sharex=True)
    for axis, (title, heading, obs_col, color, ylim) in zip(
        axes.flat, panels
    ):
        model = outlet[:, headings.index(heading)]
        valid = observations[:, obs_col] > -9999
        axis.plot(
            observations[valid, 0],
            observations[valid, obs_col],
            ".",
            color=color,
            markersize=5,
            label="Appelo et al.",
        )
        axis.plot(
            volume_ml,
            model,
            "-",
            color=color,
            linewidth=1.5,
            label="MF6PQC",
        )
        axis.plot(
            reference_volume_ml,
            reference[heading],
            "--",
            color="black",
            linewidth=1.0,
            label="PHT3D",
        )
        axis.set_title(title, pad=2)
        axis.set_xlim(-100, 800)
        axis.set_ylim(*ylim)
        axis.set_ylabel("mol/L" if heading != "pH" else "")
        axis.tick_params(direction="in", top=True, right=True)
    axes[2, 0].set_xlabel("ml outflow")
    axes[2, 1].set_xlabel("ml outflow")
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axes
