"""Render the six-panel PHT3D Example 11 comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

from validate import CASE_DIR, PROFILE_X, grid_centres, load_case_fields


PLOT_SPECS = (
    ("Toluene", "Toluene", 1.0e4, (0.0, 10.0), [0, 5, 10], r"$\times 10^{-4}$"),
    (
        "delta13C Toluene",
        r"$\delta^{13}$C Toluene",
        1.0,
        (-25.0, -18.0),
        [-24, -22, -20, -18],
        None,
    ),
    (
        "Naphthalene",
        "Naphthalene",
        1.0e4,
        (0.0, 4.0),
        [0, 1, 2, 3, 4],
        r"$\times 10^{-4}$",
    ),
    (
        "delta13C Naphthalene",
        r"$\delta^{13}$C Naphthalene",
        1.0,
        (-25.0, -18.0),
        [-24, -22, -20, -18],
        None,
    ),
    ("Sulfate", "Sulfate", 1.0e3, (0.0, 3.0), [0, 1, 2, 3], r"$\times 10^{-3}$"),
    (
        "delta34S Sulfate",
        r"$\delta^{34}$S Sulfate",
        1.0,
        (10.0, 60.0),
        [10, 20, 30, 40, 50, 60],
        None,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=CASE_DIR / "output" / "PHT3D_E11_comparison.png",
        help="PNG path (default: output/PHT3D_E11_comparison.png)",
    )
    return parser.parse_args()


def add_case_annotations(ax: plt.Axes) -> None:
    """Add the two NAPL sources and HR-MLW location."""
    source_style = {
        "fill": False,
        "edgecolor": "#ffef00",
        "linewidth": 1.3,
        "zorder": 8,
    }
    ax.add_patch(Rectangle((9.0, 33.5), 2.0, 0.3, **source_style))
    ax.add_patch(Rectangle((9.0, 33.0), 1.0, 0.05, **source_style))
    ax.axvline(PROFILE_X, color="white", linestyle=":", linewidth=1.25, zorder=7)
    ax.text(5.0, 33.22, "Sources", fontsize=9, color="black", zorder=9)
    ax.text(18.0, 32.93, "HR-MLW", fontsize=9, color="black", zorder=9)


def main() -> None:
    args = parse_args()
    times, _, model_fields, reference_fields = load_case_fields()
    x, z, x_grid, z_grid = grid_centres()
    well_column = int(np.argmin(np.abs(x - PROFILE_X)))
    wet = np.isfinite(reference_fields["Sulfate"][-1])
    profile_layers = (
        (z >= 32.5) & (z <= 34.0) & wet[:, well_column]
    )
    profile_markers = profile_layers & (np.arange(z.size) % 2 == 0)

    fig, axes = plt.subplots(
        3, 2, figsize=(11.2, 8.2), constrained_layout=True
    )
    profile_width = 6.0
    for panel, (ax, spec) in enumerate(zip(axes.flat, PLOT_SPECS)):
        name, title, scale, limits, ticks, exponent = spec
        low, high = limits
        reproduced = model_fields[name][-1] * scale
        official = reference_fields[name][-1] * scale
        displayed = np.clip(reproduced, low, high).astype(float)
        displayed[~wet] = np.nan
        contour = ax.contourf(
            x_grid,
            z_grid,
            displayed,
            levels=np.linspace(low, high, 25),
            cmap="jet",
        )

        official_profile = np.clip(official[:, well_column], low, high)
        model_profile = np.clip(reproduced[:, well_column], low, high)
        official_profile_x = (
            PROFILE_X
            + profile_width * (official_profile - low) / (high - low)
        )
        model_profile_x = (
            PROFILE_X + profile_width * (model_profile - low) / (high - low)
        )
        ax.plot(
            official_profile_x[profile_layers],
            z[profile_layers],
            color="white",
            linewidth=2.2,
            zorder=9,
        )
        ax.plot(
            model_profile_x[profile_markers],
            z[profile_markers],
            linestyle="none",
            marker="o",
            markersize=3.2,
            markerfacecolor="none",
            markeredgecolor="red",
            markeredgewidth=0.8,
            zorder=10,
        )
        add_case_annotations(ax)

        ax.set_title(title, fontsize=12, fontweight="semibold", pad=8)
        ax.set_xlim(4.0, 36.0)
        ax.set_ylim(32.5, 34.15)
        ax.set_xticks(np.arange(5, 36, 5))
        ax.set_yticks([32.5, 33.0, 33.5, 34.0])
        ax.grid(
            True,
            color="0.25",
            linestyle=":",
            linewidth=0.65,
            alpha=0.65,
        )
        ax.tick_params(direction="in", labelsize=9)
        if panel // 2 == 2:
            ax.set_xlabel("Distance (m)", fontsize=10)
        if panel % 2 == 0:
            ax.set_ylabel("Depth (m asl)", fontsize=10)

        colorbar = fig.colorbar(
            contour, ax=ax, ticks=ticks, fraction=0.050, pad=0.045
        )
        colorbar.ax.tick_params(labelsize=9)
        if exponent:
            colorbar.ax.set_title(exponent, fontsize=9, pad=5)

    legend_handles = (
        Line2D(
            [0],
            [0],
            color="0.2",
            linewidth=2.2,
            label="PHT3D HR-MLW profile",
        ),
        Line2D(
            [0],
            [0],
            linestyle="none",
            marker="o",
            markersize=4,
            markerfacecolor="none",
            markeredgecolor="red",
            label="MF6PQC HR-MLW profile",
        ),
    )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(f"PHT3D Example 11 comparison at {times[-1]:g} days", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
