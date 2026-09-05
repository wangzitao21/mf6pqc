"""Create engineering summaries from a completed SaltLake_Brine3D run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import flopy
import numpy as np

from case_config import PROFILES

CASE_DIR = Path(__file__).resolve().parent
PRIMARY_MINERALS = (
    "Halite",
    "Carnallite",
    "Polyhalite",
    "Sylvite",
    "Gypsum",
    "Borax",
)
SECONDARY_MINERALS = ("Bischofite", "Syngenite", "Mirabilite")
WELL_CHEMISTRY_FIELDS = ("K", "Li", "Na", "Mg", "Ca", "Cl", "S_6", "B", "Br")


def run_label(profile: str, scenario: str, constant_density: bool) -> str:
    label = f"{profile}_{scenario}"
    if constant_density:
        label += "_constant_density"
    return label


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _first_threshold_time(
    times_years: np.ndarray, values: np.ndarray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values >= threshold)
    return None if indices.size == 0 else float(times_years[indices[0]])


def summarize_run(output_dir: Path, simulation_dir: Path) -> dict[str, Any]:
    """Write well-quality, mineral-inventory, and feedback summary files."""
    output_dir = Path(output_dir)
    simulation_dir = Path(simulation_dir)
    metadata = json.loads(
        (output_dir / "case_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (output_dir / "results_manifest.json").read_text(encoding="utf-8")
    )
    results = np.load(output_dir / "results.npy")
    times_days = np.load(output_dir / "results_times.npy")
    headings = (output_dir / "results_headings.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    field_index = {name: index for index, name in enumerate(headings)}
    times_years = times_days / metadata["time"]["days_per_year"]
    well_indices = np.asarray(
        metadata["boundaries"]["well_flat_indices"], dtype=int
    )

    def field(name: str) -> np.ndarray:
        return results[:, field_index[name], :]

    well_fields = {
        name: np.mean(field(name)[:, well_indices], axis=1)
        for name in WELL_CHEMISTRY_FIELDS
    }
    initial_br = float(metadata["initial_selected_output"]["Br"])
    channel_br = float(metadata["channel_components_mol_per_litre"]["Br"])
    tracer_denominator = channel_br - initial_br
    if tracer_denominator <= 0.0:
        raise ValueError("Channel Br must exceed formation-brine Br")
    tracer_fraction = np.clip(
        (well_fields["Br"] - initial_br) / tracer_denominator, 0.0, 1.0
    )

    well_rows: list[dict[str, Any]] = []
    initial_k = float(metadata["initial_selected_output"]["K"])
    initial_li = float(metadata["initial_selected_output"]["Li"])
    for index, time_years in enumerate(times_years):
        row: dict[str, Any] = {
            "time_years": float(time_years),
            "tracer_fraction": float(tracer_fraction[index]),
            "K_enrichment_vs_initial": float(well_fields["K"][index] / initial_k),
            "Li_enrichment_vs_initial": float(well_fields["Li"][index] / initial_li),
        }
        for name in WELL_CHEMISTRY_FIELDS:
            row[f"{name}_mol_per_L"] = float(well_fields[name][index])
        well_rows.append(row)
    _write_csv(output_dir / "well_quality.csv", well_rows)

    cell_volumes_litres = np.load(output_dir / "cell_volumes_m3.npy") * 1000.0
    mineral_rows: list[dict[str, Any]] = []
    for mineral in (*PRIMARY_MINERALS, *SECONDARY_MINERALS):
        amounts = field(mineral)
        total_moles = np.sum(amounts * cell_volumes_litres[None, :], axis=1)
        initial_moles = float(total_moles[0])
        for index, time_years in enumerate(times_years):
            change = float(total_moles[index] - initial_moles)
            mineral_rows.append(
                {
                    "time_years": float(time_years),
                    "mineral": mineral,
                    "inventory_mol": float(total_moles[index]),
                    "change_from_initial_mol": change,
                    "fractional_change": (
                        change / initial_moles if initial_moles > 0.0 else ""
                    ),
                }
            )
    _write_csv(output_dir / "mineral_inventory.csv", mineral_rows)

    initial_porosity = np.load(output_dir / "initial_porosity.npy")
    initial_k_field = np.load(output_dir / "initial_K.npy")
    if manifest["has_porosity_and_k"]:
        porosity = np.load(output_dir / "results_porosity.npy")
        conductivity = np.load(output_dir / "results_K.npy")
    else:
        porosity = np.broadcast_to(initial_porosity, (times_days.size, initial_porosity.size))
        conductivity = np.broadcast_to(
            initial_k_field, (times_days.size, initial_k_field.size)
        )

    porosity_change = porosity[-1] - porosity[0]
    conductivity_ratio = conductivity[-1] / conductivity[0]
    positive_k_growth = np.maximum(conductivity[-1] - conductivity[0], 0.0)
    if np.any(positive_k_growth > 0.0):
        threshold = np.quantile(positive_k_growth, 0.90)
        top_growth_fraction = float(
            positive_k_growth[positive_k_growth >= threshold].sum()
            / positive_k_growth.sum()
        )
    else:
        top_growth_fraction = 0.0

    cell_volumes_m3 = cell_volumes_litres / 1000.0
    reacted_mask = porosity_change > 1.0e-4
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "profile": metadata["profile"],
        "scenario": metadata["scenario"],
        "density_feedback": metadata["density_feedback"],
        "bromide_breakthrough_5pct_years": _first_threshold_time(
            times_years, tracer_fraction, 0.05
        ),
        "bromide_breakthrough_50pct_years": _first_threshold_time(
            times_years, tracer_fraction, 0.50
        ),
        "peak_well_K_mol_per_L": float(np.max(well_fields["K"])),
        "peak_well_K_time_years": float(
            times_years[int(np.argmax(well_fields["K"]))]
        ),
        "peak_well_K_enrichment_vs_initial": float(
            np.max(well_fields["K"]) / initial_k
        ),
        "peak_well_Li_mol_per_L": float(np.max(well_fields["Li"])),
        "peak_well_Li_enrichment_vs_initial": float(
            np.max(well_fields["Li"]) / initial_li
        ),
        "maximum_porosity_increase": float(np.max(porosity_change)),
        "maximum_K_multiplier": float(np.max(conductivity_ratio)),
        "K_multiplier_p95_over_median": float(
            np.quantile(conductivity_ratio, 0.95) / np.median(conductivity_ratio)
        ),
        "top_10pct_cells_share_of_positive_K_growth": top_growth_fraction,
        "reacted_bulk_volume_m3_phi_delta_gt_1e-4": float(
            cell_volumes_m3[reacted_mask].sum()
        ),
        "reacted_cell_fraction_phi_delta_gt_1e-4": float(np.mean(reacted_mask)),
    }

    head_path = simulation_dir / "gwf_model.hds"
    budget_path = simulation_dir / "gwf_model.bud"
    if head_path.is_file() and budget_path.is_file():
        heads = np.asarray(flopy.utils.HeadFile(head_path).get_alldata())
        final_heads = heads[-1].ravel()
        metrics["minimum_final_head_m"] = float(np.min(final_heads))
        metrics["mean_final_well_head_m"] = float(
            np.mean(final_heads[well_indices])
        )
        budget = flopy.utils.CellBudgetFile(budget_path, precision="double")
        final_time = budget.get_times()[-1]
        ghb = budget.get_data(text="GHB", totim=final_time)[0]
        wells = budget.get_data(text="WEL", totim=final_time)[0]
        metrics["final_channel_inflow_m3_per_day"] = float(np.sum(ghb["q"]))
        metrics["final_well_flow_m3_per_day"] = float(np.sum(wells["q"]))

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics


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
    metrics = summarize_run(
        CASE_DIR / "output" / label,
        CASE_DIR / "simulation" / label,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
