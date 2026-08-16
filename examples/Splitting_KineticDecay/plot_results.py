"""Create the two validation figures after ``run.py`` has finished."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXAMPLE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXAMPLE_DIR / "output"
METHOD_COLORS = {
    "SNIA": "#4c78a8",
    "Strang": "#f58518",
    "SIA": "#54a24b",
}


def profile_key(method: str, cfl: float) -> str:
    token = format(cfl, "g").replace(".", "p")
    return f"profile__{method}__cfl_{token}"


def load_results():
    data_path = OUTPUT_DIR / "paper_figure6_data.npz"
    metrics_path = OUTPUT_DIR / "paper_figure6_metrics.json"
    if not data_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            "Run examples/Splitting_KineticDecay/run.py before plotting"
        )
    archive = np.load(data_path)
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    lookup = {(row["method"], float(row["cfl"])): row for row in rows}
    return archive, lookup


def plot_profiles(archive) -> Path:
    figure, axis = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    axis.plot(
        archive["x_analytical_dense_m"],
        archive["analytical_dense"],
        color="black",
        linewidth=2.4,
        label="Analytical solution",
    )
    x = archive["x_paper_nodes_m"]
    for method, marker in (("SNIA", "o"), ("Strang", "s"), ("SIA", "^")):
        axis.plot(
            x,
            archive[profile_key(method, 1.0)],
            color=METHOD_COLORS[method],
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=f"{method}, CFL=1",
        )
    axis.plot(
        x,
        archive[profile_key("SNIA", 0.1)],
        color=METHOD_COLORS["SNIA"],
        linestyle="--",
        marker=".",
        linewidth=1.5,
        label="SNIA, CFL=0.1 (refined step)",
    )
    axis.set(
        xlabel="Distance, x (m)",
        ylabel="Normalized concentration, C/C0",
        title="First-order decay: accuracy at the published CFL=1 step",
        xlim=(0.0, 6.0),
        ylim=(-0.015, 1.02),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=True)
    path = OUTPUT_DIR / "sia_validation_profiles.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def plot_accuracy_cost(lookup) -> Path:
    methods = ["SNIA", "Strang", "SIA"]
    rows = [lookup[(method, 1.0)] for method in methods]
    colors = [METHOD_COLORS[method] for method in methods]
    rmse = [row["paper_node_rmse"] for row in rows]
    solves = [row["transport_solves"] for row in rows]
    wall_time = [row["wall_time_seconds"] for row in rows]

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), constrained_layout=True)
    for axis, values, title, ylabel, log_scale in (
        (axes[0], rmse, "Accuracy (lower is better)", "RMSE", True),
        (
            axes[1],
            solves,
            "Deterministic coupling work",
            "MODFLOW transport solves",
            True,
        ),
        (
            axes[2],
            wall_time,
            "Measured runtime",
            "Wall time (s)",
            False,
        ),
    ):
        bars = axis.bar(methods, values, color=colors, width=0.68)
        if log_scale:
            axis.set_yscale("log")
        axis.set(title=title, ylabel=ylabel)
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            label = f"{value:.3g}"
            axis.annotate(
                label,
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    figure.suptitle(
        "CFL=1: accuracy improves SNIA -> Strang -> SIA; cost reverses",
        fontsize=13,
    )
    path = OUTPUT_DIR / "sia_accuracy_cost.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def main() -> None:
    archive, lookup = load_results()
    try:
        paths = (plot_profiles(archive), plot_accuracy_cost(lookup))
    finally:
        archive.close()
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
