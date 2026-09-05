"""Plot flow, chemistry, dissolution, and feedback diagnostics for one run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import flopy
import matplotlib.pyplot as plt
import numpy as np

from case_config import PROFILES
from analyze_results import run_label
from case_config import MINERAL_MOLAR_VOLUMES_L_PER_MOL


CASE_DIR = Path(__file__).resolve().parent
PRIMARY_MINERALS = (
    "Halite",
    "Carnallite",
    "Polyhalite",
    "Sylvite",
    "Gypsum",
    "Borax",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="highres")
    parser.add_argument(
        "--scenario", choices=("feedback", "fixed"), default="feedback"
    )
    parser.add_argument("--constant-density", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    label = run_label(args.profile, args.scenario, args.constant_density)
    output_dir = CASE_DIR / "output" / label
    simulation_dir = CASE_DIR / "simulation" / label
    metadata = json.loads(
        (output_dir / "case_metadata.json").read_text(encoding="utf-8")
    )
    results = np.load(output_dir / "results.npy")
    times = np.load(output_dir / "results_times.npy") / 365.0
    headings = (output_dir / "results_headings.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    field_index = {name: index for index, name in enumerate(headings)}
    shape = (
        metadata["grid"]["nlay"],
        metadata["grid"]["nrow"],
        metadata["grid"]["ncol"],
    )
    extent = (
        0.0,
        shape[2] * metadata["grid"]["delr_m"],
        0.0,
        shape[1] * metadata["grid"]["delc_m"],
    )
    well_indices = np.asarray(
        metadata["boundaries"]["well_flat_indices"], dtype=int
    )

    heads = flopy.utils.HeadFile(simulation_dir / "gwf_model.hds").get_alldata()
    final_head = np.asarray(heads[-1])[0]
    final_k_aqueous = results[-1, field_index["K"], :].reshape(shape)[0]

    initial_porosity = np.load(output_dir / "initial_porosity.npy")
    initial_conductivity = np.load(output_dir / "initial_K.npy")
    porosity_path = output_dir / "results_porosity.npy"
    conductivity_path = output_dir / "results_K.npy"
    if porosity_path.is_file() and conductivity_path.is_file():
        porosity = np.load(porosity_path)
        conductivity = np.load(conductivity_path)
        porosity_change = (porosity[-1] - porosity[0]).reshape(shape)[0]
        conductivity_ratio = (conductivity[-1] / conductivity[0]).reshape(shape)[0]
    else:
        porosity_change = np.zeros(shape)[0]
        conductivity_ratio = np.ones(shape)[0]

    dissolved_volume_fraction = np.zeros(np.prod(shape), dtype=float)
    for mineral in PRIMARY_MINERALS:
        initial_amount = results[0, field_index[mineral], :]
        final_amount = results[-1, field_index[mineral], :]
        dissolved_volume_fraction += (
            initial_amount - final_amount
        ) * MINERAL_MOLAR_VOLUMES_L_PER_MOL[mineral]
    dissolved_volume_fraction = dissolved_volume_fraction.reshape(shape)[0]

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    panels = (
        (axes[0, 0], final_head, "Final head, layer 1 (m)", "viridis"),
        (axes[0, 1], final_k_aqueous, "Final dissolved K, layer 1 (mol/L)", "plasma"),
        (
            axes[0, 2],
            dissolved_volume_fraction,
            "Primary-mineral volume dissolved, layer 1",
            "magma",
        ),
        (axes[1, 0], porosity_change, "Porosity change, layer 1", "cividis"),
        (axes[1, 1], conductivity_ratio, "K/K0, layer 1", "inferno"),
    )
    for axis, values, title, cmap in panels:
        image = axis.imshow(
            values,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        figure.colorbar(image, ax=axis, shrink=0.82)
        axis.set_title(title)
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        channel_cells = metadata["boundaries"]["channel_cells"]
        well_cells = metadata["boundaries"]["well_cells"]
        axis.scatter(
            [(cell[2] + 0.5) * metadata["grid"]["delr_m"] for cell in channel_cells],
            [(cell[1] + 0.5) * metadata["grid"]["delc_m"] for cell in channel_cells],
            marker="s",
            s=18,
            facecolors="none",
            edgecolors="cyan",
            label="channel",
        )
        axis.scatter(
            [(cell[2] + 0.5) * metadata["grid"]["delr_m"] for cell in well_cells],
            [(cell[1] + 0.5) * metadata["grid"]["delc_m"] for cell in well_cells],
            marker="v",
            s=28,
            color="white",
            edgecolors="black",
            label="well",
        )

    curve_axis = axes[1, 2]
    well_k = np.mean(results[:, field_index["K"], :][:, well_indices], axis=1)
    well_li = np.mean(results[:, field_index["Li"], :][:, well_indices], axis=1)
    well_br = np.mean(results[:, field_index["Br"], :][:, well_indices], axis=1)
    br_initial = metadata["initial_selected_output"]["Br"]
    br_channel = metadata["channel_components_mol_per_litre"]["Br"]
    tracer = np.clip((well_br - br_initial) / (br_channel - br_initial), 0.0, 1.0)
    curve_axis.plot(times, well_k, "-o", label="K (mol/L)")
    curve_axis.plot(times, 100.0 * well_li, "-s", label="100 x Li (mol/L)")
    curve_axis.set_xlabel("time (years)")
    curve_axis.set_ylabel("well concentration")
    curve_axis.set_title("Flux-weighted production-well chemistry")
    tracer_axis = curve_axis.twinx()
    tracer_axis.plot(times, tracer, "--", color="black", label="Br fraction")
    tracer_axis.set_ylabel("channel-water fraction from Br")
    tracer_axis.set_ylim(0.0, 1.0)
    lines = curve_axis.lines + tracer_axis.lines
    curve_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    figure.suptitle(f"SaltLake_Brine3D: {label}")
    destination = output_dir / "overview.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
