"""Reproduce Implicit validation without overwriting example outputs.

Run from the repository root:
    python scripts/validate_implicit.py --output-root tmp/implicit_validation
Requires the example dependencies and the configured MODFLOW native library.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]


def run_b1(root, step):
    target = root / ("b1_coarse" if step == 0.3 else "b1_refined")
    env = dict(os.environ, MF6PQC_RUN_ROOT=str(target), OPENBLAS_NUM_THREADS="1")
    with (root / (target.name + ".log")).open("w", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "examples/ex014_Xie2015_B1/run.py"),
                "--years",
                "500",
                "--max-step-years",
                str(step),
            ],
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    output = target / "ex014_Xie2015_B1/output"
    manifest = json.loads((output / "results_manifest.json").read_text())
    return output, manifest["run"]


def compare_b1(coarse, refined):
    times = [np.load(p / "results_times.npy") for p in (coarse, refined)]
    porosity = [np.load(p / "results_porosity.npy") for p in (coarse, refined)]
    results = [np.load(p / "results.npy") for p in (coarse, refined)]
    headings = (coarse / "results_headings.txt").read_text().splitlines()
    reference = np.genfromtxt(
        ROOT / "examples/ex014_Xie2015_B1/input_data/MIN3P_results.csv",
        delimiter=",",
        names=True,
    )
    widths = np.array([0.0125] + [0.025] * 79 + [0.0125])
    x = np.cumsum(widths) - widths / 2
    report = {}
    for year in (10, 100, 120, 500):
        indices = [int(np.argmin(abs(t - year * 365))) for t in times]
        if any(abs(t[i] - year * 365) > 1e-6 for t, i in zip(times, indices, strict=True)):
            raise AssertionError("Comparison time is absent from saved outputs")
        a, b = indices
        row = {
            "porosity_max_step_difference": float(np.max(abs(porosity[0][a] - porosity[1][b]))),
            "calcite_mol_bulk_max_step_difference": float(
                np.max(
                    abs(
                        results[0][a, headings.index("Calcite")]
                        - results[1][b, headings.index("Calcite")]
                    )
                )
            ),
        }
        key = f"Porosity_{year}years"
        if key in reference.dtype.names:
            delta = np.interp(np.linspace(0, 2, len(reference)), x, porosity[0][a]) - reference[key]
            row["porosity_reference_rmse"] = float(np.sqrt(np.mean(delta**2)))
        report[str(year)] = row
    return report


def main():
    from tests.test_implicit_native import calculate

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-b1", action="store_true", help="Only run the native 3-D comparison")
    args = parser.parse_args()
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    report = {}
    if not args.skip_b1:
        coarse, metadata = run_b1(root, 0.3)
        refined, refined_metadata = run_b1(root, 0.075)
        report["b1"] = {
            "coarse": metadata,
            "refined": refined_metadata,
            "errors": compare_b1(coarse, refined),
        }
        print("B1 refinement complete.", flush=True)
    runs = {}
    for label, method, steps, dense_limit in (
        ("implicit", "Implicit", 100, 400),
        ("matrix_free", "Implicit", 100, 1),
        ("native", "SNIA", 1000, 400),
    ):
        values = calculate(root / label, method, steps, dense_limit=dense_limit)
        runs[label] = values
        np.savez(
            root / (label + ".npz"),
            minerals=values[0],
            concentrations=values[1],
            component_budget=values[2],
        )
        print(label, "seconds", values[3], flush=True)
    native = runs["native"]
    report["native_3d"] = {}
    for label, values in runs.items():
        record = {
            "wall_time_seconds": values[3],
            "max_mineral_error": float(np.max(abs(values[0] - native[0]))),
            "max_aqueous_error": float(np.max(abs(values[1] - native[1]))),
            "max_closed_component_budget": float(np.max(abs(values[2]))),
        }
        report["native_3d"][label] = record
        if label != "native":
            assert record["max_mineral_error"] < 2e-5
            assert record["max_aqueous_error"] < 2e-4
            assert record["max_closed_component_budget"] < 1e-7
    report["native_3d"]["dense_matrix_free_max_mineral_difference"] = float(
        np.max(abs(runs["implicit"][0] - runs["matrix_free"][0]))
    )
    assert report["native_3d"]["dense_matrix_free_max_mineral_difference"] < 1e-8
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(root / "validation.json")


if __name__ == "__main__":
    main()
