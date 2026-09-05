"""Compare feedback and fixed-property runs on the same profile."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from analyze_results import run_label, summarize_run
from case_config import PROFILES


CASE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="highres")
    parser.add_argument("--constant-density", action="store_true")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    summaries = {}
    quality = {}
    for scenario in ("feedback", "fixed"):
        label = run_label(args.profile, scenario, args.constant_density)
        output_dir = CASE_DIR / "output" / label
        simulation_dir = CASE_DIR / "simulation" / label
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

    suffix = "_constant_density" if args.constant_density else ""
    comparison_csv = CASE_DIR / "output" / (
        f"{args.profile}_feedback_vs_fixed{suffix}.csv"
    )
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)

    comparison = {
        "profile": args.profile,
        "constant_density": args.constant_density,
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


if __name__ == "__main__":
    main()
