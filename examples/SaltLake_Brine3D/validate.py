"""Numerical and scientific-contract checks for a completed salt-lake run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from analyze_results import run_label, summarize_run
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
    manifest = json.loads(
        (output_dir / "results_manifest.json").read_text(encoding="utf-8")
    )
    results = np.load(output_dir / "results.npy")
    times = np.load(output_dir / "results_times.npy")
    headings = (output_dir / "results_headings.txt").read_text(
        encoding="utf-8"
    ).splitlines()
    index = {name: position for position, name in enumerate(headings)}

    expected_frames = metadata["time"]["years"] + 1
    expected_cells = (
        metadata["grid"]["nlay"]
        * metadata["grid"]["nrow"]
        * metadata["grid"]["ncol"]
    )
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

    listing = (simulation_dir / "gwf_model.lst").read_text(
        encoding="utf-8", errors="replace"
    )
    discrepancies = [
        abs(float(value))
        for value in re.findall(r"PERCENT DISCREPANCY\s*=\s*([-+0-9.Ee]+)", listing)
    ]
    if not discrepancies or max(discrepancies) > 0.10:
        raise AssertionError(
            f"Flow-budget percent discrepancy is unacceptable: {discrepancies}"
        )
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
    main()
