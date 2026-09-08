"""Numerical and scientific-contract checks for a completed salt-lake run."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import json
import re

import _example_support as _example_support
import numpy as np
from _example_support import runtime_path
from modflow_model import PROFILES

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


import csv
from typing import Any

import flopy

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
    metadata = json.loads((output_dir / "case_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "results_manifest.json").read_text(encoding="utf-8"))
    results = np.load(output_dir / "results.npy")
    times_days = np.load(output_dir / "results_times.npy")
    headings = (output_dir / "results_headings.txt").read_text(encoding="utf-8").splitlines()
    field_index = {name: index for index, name in enumerate(headings)}
    times_years = times_days / metadata["time"]["days_per_year"]
    well_indices = np.asarray(metadata["boundaries"]["well_flat_indices"], dtype=int)

    def field(name: str) -> np.ndarray:
        return results[:, field_index[name], :]

    well_fields = {
        name: np.mean(field(name)[:, well_indices], axis=1) for name in WELL_CHEMISTRY_FIELDS
    }
    initial_br = float(metadata["initial_selected_output"]["Br"])
    channel_br = float(metadata["channel_components_mol_per_litre"]["Br"])
    tracer_denominator = channel_br - initial_br
    if tracer_denominator <= 0.0:
        raise ValueError("Channel Br must exceed formation-brine Br")
    tracer_fraction = np.clip((well_fields["Br"] - initial_br) / tracer_denominator, 0.0, 1.0)

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
                    "fractional_change": (change / initial_moles if initial_moles > 0.0 else ""),
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
        conductivity = np.broadcast_to(initial_k_field, (times_days.size, initial_k_field.size))

    porosity_change = porosity[-1] - porosity[0]
    conductivity_ratio = conductivity[-1] / conductivity[0]
    positive_k_growth = np.maximum(conductivity[-1] - conductivity[0], 0.0)
    if np.any(positive_k_growth > 0.0):
        threshold = np.quantile(positive_k_growth, 0.90)
        top_growth_fraction = float(
            positive_k_growth[positive_k_growth >= threshold].sum() / positive_k_growth.sum()
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
        "peak_well_K_time_years": float(times_years[int(np.argmax(well_fields["K"]))]),
        "peak_well_K_enrichment_vs_initial": float(np.max(well_fields["K"]) / initial_k),
        "peak_well_Li_mol_per_L": float(np.max(well_fields["Li"])),
        "peak_well_Li_enrichment_vs_initial": float(np.max(well_fields["Li"]) / initial_li),
        "maximum_porosity_increase": float(np.max(porosity_change)),
        "maximum_K_multiplier": float(np.max(conductivity_ratio)),
        "K_multiplier_p95_over_median": float(
            np.quantile(conductivity_ratio, 0.95) / np.median(conductivity_ratio)
        ),
        "top_10pct_cells_share_of_positive_K_growth": top_growth_fraction,
        "reacted_bulk_volume_m3_phi_delta_gt_1e-4": float(cell_volumes_m3[reacted_mask].sum()),
        "reacted_cell_fraction_phi_delta_gt_1e-4": float(np.mean(reacted_mask)),
    }

    head_path = simulation_dir / "gwf_model.hds"
    budget_path = simulation_dir / "gwf_model.bud"
    if head_path.is_file() and budget_path.is_file():
        heads = np.asarray(flopy.utils.HeadFile(head_path).get_alldata())
        final_heads = heads[-1].ravel()
        metrics["minimum_final_head_m"] = float(np.min(final_heads))
        metrics["mean_final_well_head_m"] = float(np.mean(final_heads[well_indices]))
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare_scenarios(profile: str, constant_density: bool = False) -> None:
    summaries = {}
    quality = {}
    for scenario in ("feedback", "fixed"):
        label = run_label(profile, scenario, constant_density)
        output_dir = runtime_path(__file__, "output") / label
        simulation_dir = runtime_path(__file__, "simulation") / label
        summaries[scenario] = summarize_run(output_dir, simulation_dir)
        quality[scenario] = _read_csv(output_dir / "well_quality.csv")

    if len(quality["feedback"]) != len(quality["fixed"]):
        raise ValueError("Feedback and fixed runs have different output schedules")
    curve_rows = []
    for feedback, fixed in zip(quality["feedback"], quality["fixed"], strict=True):
        if feedback["time_years"] != fixed["time_years"]:
            raise ValueError("Feedback and fixed output times do not match")
        curve_rows.append(
            {
                "time_years": feedback["time_years"],
                "K_feedback_mol_per_L": feedback["K_mol_per_L"],
                "K_fixed_mol_per_L": fixed["K_mol_per_L"],
                "K_feedback_minus_fixed_mol_per_L": (
                    float(feedback["K_mol_per_L"]) - float(fixed["K_mol_per_L"])
                ),
                "Li_feedback_mol_per_L": feedback["Li_mol_per_L"],
                "Li_fixed_mol_per_L": fixed["Li_mol_per_L"],
                "tracer_feedback": feedback["tracer_fraction"],
                "tracer_fixed": fixed["tracer_fraction"],
            }
        )

    suffix = "_constant_density" if constant_density else ""
    comparison_csv = runtime_path(__file__, "output") / (f"{profile}_feedback_vs_fixed{suffix}.csv")
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    comparison = {
        "profile": profile,
        "constant_density": constant_density,
        "feedback": summaries["feedback"],
        "fixed": summaries["fixed"],
        "differences": {
            "peak_well_K_mol_per_L": (
                summaries["feedback"]["peak_well_K_mol_per_L"]
                - summaries["fixed"]["peak_well_K_mol_per_L"]
            ),
            "maximum_K_multiplier": (
                summaries["feedback"]["maximum_K_multiplier"]
                - summaries["fixed"]["maximum_K_multiplier"]
            ),
            "mean_final_well_head_m": (
                summaries["feedback"]["mean_final_well_head_m"]
                - summaries["fixed"]["mean_final_well_head_m"]
            ),
        },
    }
    comparison_json = comparison_csv.with_suffix(".json")
    comparison_json.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {comparison_csv}")
    print(f"Saved {comparison_json}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="highres")
    parser.add_argument("--scenario", choices=("feedback", "fixed"), default="feedback")
    parser.add_argument("--constant-density", action="store_true")
    parser.add_argument(
        "--compare", action="store_true", help="compare feedback and fixed scenarios"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare:
        compare_scenarios(args.profile, args.constant_density)
        return
    label = run_label(args.profile, args.scenario, args.constant_density)
    output_dir = runtime_path(__file__, "output") / label
    simulation_dir = runtime_path(__file__, "simulation") / label

    metadata = json.loads((output_dir / "case_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "results_manifest.json").read_text(encoding="utf-8"))
    results = np.load(output_dir / "results.npy")
    times = np.load(output_dir / "results_times.npy")
    headings = (output_dir / "results_headings.txt").read_text(encoding="utf-8").splitlines()
    index = {name: position for position, name in enumerate(headings)}

    expected_frames = metadata["time"]["years"] + 1
    expected_cells = metadata["grid"]["nlay"] * metadata["grid"]["nrow"] * metadata["grid"]["ncol"]
    if results.shape != (expected_frames, len(headings), expected_cells):
        raise AssertionError(f"Unexpected result shape: {results.shape}")
    if times.shape != (expected_frames,) or not np.all(np.diff(times) > 0.0):
        raise AssertionError("Saved times are incomplete or non-monotonic")
    if not np.all(np.isfinite(results)):
        raise AssertionError("Selected output contains non-finite values")

    required = {
        "Na",
        "K",
        "Li",
        "Mg",
        "Ca",
        "Cl",
        "S_6",
        "B",
        "Br",
        "RHO",
        *(PRIMARY_MINERALS + SECONDARY_MINERALS),
    }
    missing = required - set(headings)
    if missing:
        raise AssertionError(f"Missing selected-output fields: {sorted(missing)}")

    for mineral in (*PRIMARY_MINERALS, *SECONDARY_MINERALS):
        minimum = float(np.min(results[:, index[mineral], :]))
        if minimum < -1.0e-9:
            raise AssertionError(f"{mineral} has negative inventory: {minimum}")

    if manifest["run"]["modflow_convergence_failures"]:
        raise AssertionError("MODFLOW convergence failures were recorded")
    if manifest["run"]["logical_steps"] != metadata["time"]["total_steps"]:
        raise AssertionError("The run did not complete every configured time step")

    if manifest["has_porosity_and_k"]:
        porosity = np.load(output_dir / "results_porosity.npy")
        conductivity = np.load(output_dir / "results_K.npy")
        if not np.all((porosity > 0.0) & (porosity < 0.95)):
            raise AssertionError("Porosity left the case acceptance range (0, 0.95)")
        if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
            raise AssertionError("Hydraulic conductivity is invalid")
        exponent = float(metadata.get("permeability_exponent", 3.0))
        expected_k = conductivity[0] * (porosity / porosity[0]) ** exponent
        np.testing.assert_allclose(conductivity, expected_k, rtol=2.0e-9, atol=1.0e-12)
        if float(np.max(porosity[-1] - porosity[0])) <= 1.0e-5:
            raise AssertionError("No resolvable mineral-driven porosity change occurred")
    elif args.scenario == "feedback":
        raise AssertionError("Feedback scenario did not save porosity/K fields")

    if metadata["density_feedback"]:
        density = results[:, index["RHO"], :]
        if np.any(density <= 0.9) or np.any(density >= 1.5):
            raise AssertionError("Pitzer density is outside the brine acceptance range")
        if float(np.ptp(density[-1])) <= 1.0e-4:
            raise AssertionError("Density feedback has no spatial signal")

    listing = (simulation_dir / "gwf_model.lst").read_text(encoding="utf-8", errors="replace")
    discrepancies = [
        abs(float(value))
        for value in re.findall(r"PERCENT DISCREPANCY\s*=\s*([-+0-9.Ee]+)", listing)
    ]
    if not discrepancies or max(discrepancies) > 0.10:
        raise AssertionError(f"Flow-budget percent discrepancy is unacceptable: {discrepancies}")
    if "FAILED TO MEET SOLVER CONVERGENCE CRITERIA" in listing:
        raise AssertionError("GWF listing contains a convergence failure")

    metrics = summarize_run(output_dir, simulation_dir)
    pumping = abs(float(metrics["final_well_flow_m3_per_day"]))
    recharge = float(metrics["final_channel_inflow_m3_per_day"])
    if pumping <= 0.0 or recharge <= 0.0:
        raise AssertionError("Well extraction or channel infiltration is absent")
    if abs(recharge - pumping) / pumping > 0.02:
        raise AssertionError("Final channel inflow does not balance pumping within 2%")
    if metrics["mean_final_well_head_m"] >= metadata["boundaries"]["channel_stage_m"]:
        raise AssertionError("Pumping did not create drawdown at the production wells")

    print(
        "Validation passed: "
        f"max flow discrepancy={max(discrepancies):.3g}%, "
        f"channel inflow={recharge:.3f} m3/d, pumping={pumping:.3f} m3/d."
    )


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
