"""Loading, comparison, and plotting helpers for PHT3D Example 4."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SPECIES = ("Ca", "Cl", "Na", "K")
COLORS = {"Ca": "red", "Cl": "blue", "Na": "magenta", "K": "green"}
YLABELS = {
    "Ca": r"Ca (mmol L$^{-1}$)",
    "Cl": r"Cl (mmol L$^{-1}$)",
    "Na": r"Na (mmol L$^{-1}$)",
    "K": r"K (mmol L$^{-1}$)",
}


def load_comparison(case_dir: str | Path):
    """Load MF6PQC selected output and the extracted official outlet curves."""
    case_dir = Path(case_dir)
    results = np.load(case_dir / "output" / "results.npy")
    headings = (
        case_dir / "output" / "results_headings.txt"
    ).read_text(encoding="utf-8").splitlines()
    official = np.load(case_dir / "input_data" / "official_reference.npz")

    if results.shape[0] != official["pore_volumes"].size:
        raise ValueError(
            "MF6PQC and official result counts differ: "
            f"{results.shape[0]} != {official['pore_volumes'].size}"
        )

    mf6pqc = {
        species: results[:, headings.index(species), -1] * 1000.0
        for species in SPECIES
    }
    pht3d = {species: official[species] * 1000.0 for species in SPECIES}
    return official["pore_volumes"], mf6pqc, pht3d


def comparison_metrics(mf6pqc: dict, pht3d: dict) -> dict:
    """Return RMSE, normalized RMSE, maximum error, and correlation."""
    metrics = {}
    for species in SPECIES:
        calculated = np.asarray(mf6pqc[species])
        reference = np.asarray(pht3d[species])
        residual = calculated - reference
        scale = np.ptp(reference)
        metrics[species] = {
            "RMSE (mmol/L)": float(np.sqrt(np.mean(residual**2))),
            "NRMSE": float(np.sqrt(np.mean(residual**2)) / scale),
            "Max abs. error (mmol/L)": float(np.max(np.abs(residual))),
            "Correlation": float(np.corrcoef(calculated, reference)[0, 1]),
        }
    return metrics


def plot_official_style(pore_volumes, mf6pqc, pht3d):
    """Recreate the official E04 breakthrough figure."""
    fig, ax = plt.subplots(figsize=(12.4, 6.2))

    # Preserve the official legend ordering: four lines, then four point sets.
    for species in SPECIES:
        ax.plot(
            pore_volumes,
            mf6pqc[species],
            color=COLORS[species],
            linewidth=1.5,
            label=f"{species} (MF6PQC)",
        )
    for species in SPECIES:
        ax.plot(
            pore_volumes,
            pht3d[species],
            linestyle="none",
            marker="o",
            markersize=4.2,
            markerfacecolor=COLORS[species],
            markeredgecolor="black",
            markeredgewidth=0.35,
            label=f"{species} (PHT3D)",
        )

    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel("PV", fontsize=14)
    ax.set_ylabel(r"(mmol L$^{-1}$)", fontsize=14)
    ax.tick_params(labelsize=12, direction="in", top=True, right=True)
    ax.legend(
        loc="upper right",
        ncol=1,
        fontsize=10.5,
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
    )
    fig.tight_layout()
    return fig, ax


def plot_species_panels(pore_volumes, mf6pqc, pht3d):
    """Plot one direct comparison panel per transported cation/anion."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    for ax, species in zip(axes.flat, SPECIES):
        ax.plot(
            pore_volumes,
            mf6pqc[species],
            color=COLORS[species],
            linewidth=1.8,
            label="MF6PQC",
        )
        ax.scatter(
            pore_volumes,
            pht3d[species],
            s=13,
            color=COLORS[species],
            edgecolor="black",
            linewidth=0.3,
            label="PHT3D",
            zorder=3,
        )
        ax.set_ylabel(YLABELS[species])
        ax.set_xlim(0.0, 3.0)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[1, 0].set_xlabel("Pore volumes")
    axes[1, 1].set_xlabel("Pore volumes")
    fig.tight_layout()
    return fig, axes


def plot_residuals(pore_volumes, mf6pqc, pht3d):
    """Plot MF6PQC minus official PHT3D outlet concentrations."""
    fig, ax = plt.subplots(figsize=(10.5, 4.5))
    for species in SPECIES:
        ax.plot(
            pore_volumes,
            np.asarray(mf6pqc[species]) - np.asarray(pht3d[species]),
            color=COLORS[species],
            label=species,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlim(0.0, 3.0)
    ax.set_xlabel("Pore volumes")
    ax.set_ylabel(r"MF6PQC - PHT3D (mmol L$^{-1}$)")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, ncol=4)
    fig.tight_layout()
    return fig, ax
