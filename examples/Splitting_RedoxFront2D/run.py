from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mf6pqc import MF6PQC
from mf6pqc.utils import get_gwt_model_name
from modflow_model import (
    LENS_OXIDANT_CAPACITY,
    MATRIX_OXIDANT_CAPACITY,
    build_transport_model,
    hydraulic_conductivity_field,
    oxidant_capacity_field,
    reactive_lens_mask,
)


NROW = 10
NCOL = 20
NXYZ = NROW * NCOL
LENGTH = 8.0
WIDTH = 4.0
POROSITY = 0.35
PULSE_DURATION = 8.0
FLUSH_DURATION = 4.0
COARSE_STEPS = (1, 1)
REFERENCE_CHECK_STEPS = (32, 16)
REFERENCE_STEPS = (64, 32)
METHODS = ("SNIA", "Strang", "SIA")
REPORT_COMPONENTS = ("Don",)
FIELD_NAMES = (*REPORT_COMPONENTS, "Extent")
FIELD_SCALES = {"Don": 1.0e-3, "Extent": LENS_OXIDANT_CAPACITY}
KINETIC_RATE_PER_DAY = 5.0
SIA_MAX_ITERATIONS = 250
SIA_RTOL = 1.0e-6
SIA_ATOL = 2.0e-8
SIA_SOURCE_RELAXATION = 0.85


def _run_paths(label: str) -> tuple[Path, Path]:
    return EXAMPLE_DIR / "simulation" / label, EXAMPLE_DIR / "output" / "runs" / label


def _kinetics_zones() -> np.ndarray:
    return np.where(reactive_lens_mask(NROW, NCOL).ravel(), 2, 1).astype(np.int32)


def run_realization(
    method: str, logical_steps: tuple[int, int], label: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Run one realization in a process-isolated native backend."""
    workspace, output_dir = _run_paths(label)
    simulator = MF6PQC(
        case_name=f"redox_front_2d_{label}",
        nxyz=NXYZ,
        nthreads=2,
        porosity=POROSITY,
        saturation=1.0,
        temperature=20.0,
        db_path=str(
            REPO_ROOT
            / "examples"
            / "PHT3D_E01"
            / "input_data"
            / "phreeqc.dat"
        ),
        pqi_path=str(EXAMPLE_DIR / "input_data" / "input.pqi"),
        modflow_dll_path=str(REPO_ROOT / "bin" / "mf6.7.0" / "libmf6.dll"),
        workspace=str(workspace),
        output_dir=str(output_dir),
        progress_interval=100,
        sia_max_iterations=SIA_MAX_ITERATIONS,
        sia_rtol=SIA_RTOL,
        sia_atol=SIA_ATOL,
        sia_source_relaxation=SIA_SOURCE_RELAXATION,
        sia_fail_on_nonconvergence=True,
    )
    try:
        initial = simulator.setup(
            {"solution": 0, "kinetics": _kinetics_zones()}
        )
        pulse = simulator.get_initial_concentrations(1)
        background = simulator.get_initial_concentrations(0)
        components = simulator.get_components()
        missing = sorted(set(REPORT_COMPONENTS) - set(components))
        if missing:
            raise RuntimeError(f"PHREEQC did not expose components: {missing}")
        build_transport_model(
            str(workspace),
            components,
            initial,
            pulse,
            background,
            nrow=NROW,
            ncol=NCOL,
            length=LENGTH,
            width=WIDTH,
            porosity=POROSITY,
            pulse_duration=PULSE_DURATION,
            flush_duration=FLUSH_DURATION,
            logical_steps_per_period=logical_steps,
            strang_half_steps=method == "Strang",
        )
        getattr(simulator, f"run_{method}")()
        fields = {}
        for component in REPORT_COMPONENTS:
            model_name = get_gwt_model_name(component)
            address = simulator.modflow_api.get_var_address("X", model_name)
            fields[component] = np.asarray(
                simulator.modflow_api.get_value(address), dtype=float
            ).reshape(NROW, NCOL)
        heading_lookup = {
            heading.casefold(): index
            for index, heading in enumerate(simulator.headings)
        }
        try:
            extent_index = heading_lookup["redox_extent"]
        except KeyError as exc:
            raise RuntimeError(
                "Selected output must contain the Redox_extent heading"
            ) from exc
        fields["Extent"] = np.asarray(
            simulator.selected_output[extent_index], dtype=float
        ).reshape(NROW, NCOL)
        sia_iterations = (
            int(np.sum(simulator.sia_iterations)) if method == "SIA" else None
        )
        logical_step_count = int(sum(logical_steps))
        metadata = {
            "label": label,
            "method": method,
            "logical_steps_per_period": list(logical_steps),
            "logical_step_durations_days": [
                PULSE_DURATION / logical_steps[0],
                FLUSH_DURATION / logical_steps[1],
            ],
            "logical_steps": logical_step_count,
            "transport_solves": {
                "SNIA": logical_step_count,
                "Strang": 2 * logical_step_count,
                "SIA": sia_iterations,
            }[method],
            "reaction_evaluations": {
                "SNIA": logical_step_count,
                "Strang": logical_step_count,
                "SIA": sia_iterations,
            }[method],
            "total_sia_iterations": sia_iterations,
            "wall_time_seconds": simulator.last_run_wall_time_seconds,
            "sia_diagnostics": list(simulator.sia_diagnostics),
        }
        return fields, metadata
    finally:
        simulator.finalize()


def _persist_child(label: str, fields: dict[str, np.ndarray], metadata: dict) -> None:
    _, output_dir = _run_paths(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "final_fields.npz", **fields)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _launch(
    method: str, logical_steps: tuple[int, int], label: str
) -> tuple[dict[str, np.ndarray], dict]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--method",
        method,
        "--steps",
        str(logical_steps[0]),
        str(logical_steps[1]),
        "--label",
        label,
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    _, output_dir = _run_paths(label)
    with np.load(output_dir / "final_fields.npz") as archive:
        fields = {name: archive[name].copy() for name in FIELD_NAMES}
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return fields, metadata


def _validate_fields(label: str, fields: dict[str, np.ndarray]) -> None:
    for component, field in fields.items():
        if field.shape != (NROW, NCOL) or not np.all(np.isfinite(field)):
            raise AssertionError(f"{label}/{component} is not a finite 2-D field")
        if np.min(field) < -2.0e-10:
            raise AssertionError(f"{label}/{component} contains negative concentration")
    if np.max(fields["Don"]) > 1.003e-3:
        raise AssertionError(f"{label} donor exceeds the inlet maximum")
    capacity = oxidant_capacity_field(NROW, NCOL)
    if np.any(fields["Extent"] > capacity + 5.0e-8):
        raise AssertionError(f"{label} exceeds the local solid-oxidant capacity")
    if np.min(fields["Extent"]) < -2.0e-10:
        raise AssertionError(f"{label} has an invalid cumulative reaction extent")


def field_error_metrics(
    fields: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    normalized_differences = []
    for field_name in FIELD_NAMES:
        difference = fields[field_name] - reference[field_name]
        scale = FIELD_SCALES[field_name]
        metrics[f"{field_name}_rmse"] = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{field_name}_nrmse"] = float(
            np.sqrt(np.mean((difference / scale) ** 2))
        )
        metrics[f"{field_name}_linf"] = float(np.max(np.abs(difference)))
        normalized_differences.append((difference / scale).ravel())
    metrics["combined_nrmse"] = float(
        np.sqrt(np.mean(np.concatenate(normalized_differences) ** 2))
    )
    return metrics


def plume_diagnostics(fields: dict[str, np.ndarray]) -> dict[str, float]:
    delr = LENGTH / NCOL
    delc = WIDTH / NROW
    water_volume_litres = delr * delc * POROSITY * 1000.0
    x = (np.arange(NCOL) + 0.5) * delr
    donor = fields["Don"]
    extent = fields["Extent"]
    capacity = oxidant_capacity_field(NROW, NCOL)
    donor_mass = float(np.sum(donor) * water_volume_litres)
    donor_by_column = np.sum(donor, axis=0)
    centroid = (
        float(np.sum(donor_by_column * x) / np.sum(donor_by_column))
        if np.sum(donor_by_column) > 0.0
        else 0.0
    )
    lens = reactive_lens_mask(NROW, NCOL)
    lens_extent_fraction = (
        float(np.sum(extent[lens]) / np.sum(extent))
        if np.sum(extent) > 0.0
        else 0.0
    )
    active_area = float(
        np.count_nonzero((donor > 5.0e-5) & (extent > 1.0e-6)) * delr * delc
    )
    depleted_area = float(
        np.count_nonzero(extent / capacity >= 0.90) * delr * delc
    )
    return {
        "aqueous_donor_mol": donor_mass,
        "summed_oxidant_consumption_model_mol": float(np.sum(extent)),
        "oxidant_capacity_utilization_fraction": float(
            np.sum(extent) / np.sum(capacity)
        ),
        "donor_centroid_x_m": centroid,
        "reaction_extent_fraction_in_lens": lens_extent_fraction,
        "overlap_area_m2": active_area,
        "ninety_percent_depleted_area_m2": depleted_area,
    }


def _write_metrics(rows: list[dict], output_dir: Path) -> None:
    (output_dir / "comparison_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fieldnames = sorted({key for row in rows for key in row})
    with (output_dir / "comparison_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison() -> None:
    """Run three coarse algorithms and an independently refined reference."""
    output_dir = EXAMPLE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, dict[str, np.ndarray]] = {}
    work: dict[str, dict] = {}
    for method in METHODS:
        fields[method], work[method] = _launch(
            method, COARSE_STEPS, f"coarse_{method.lower()}"
        )
        _validate_fields(method, fields[method])
    fields["ReferenceCheck"], work["ReferenceCheck"] = _launch(
        "Strang", REFERENCE_CHECK_STEPS, "reference_check_strang"
    )
    _validate_fields("ReferenceCheck", fields["ReferenceCheck"])
    fields["Reference"], work["Reference"] = _launch(
        "Strang", REFERENCE_STEPS, "reference_strang"
    )
    _validate_fields("Reference", fields["Reference"])
    fields["ReferenceCrossCheck"], work["ReferenceCrossCheck"] = _launch(
        "SNIA", REFERENCE_STEPS, "reference_crosscheck_snia"
    )
    _validate_fields("ReferenceCrossCheck", fields["ReferenceCrossCheck"])

    rows = []
    for label in (
        *METHODS,
        "ReferenceCheck",
        "ReferenceCrossCheck",
        "Reference",
    ):
        errors = (
            field_error_metrics(fields[label], fields["Reference"])
            if label != "Reference"
            else {
                f"{component}_{metric}": 0.0
                for component in FIELD_NAMES
                for metric in (
                    "rmse",
                    "nrmse",
                    "linf",
                )
            }
        )
        errors["combined_nrmse"] = errors.get("combined_nrmse", 0.0)
        rows.append(
            {
                "label": label,
                "method": work[label]["method"],
                **errors,
                **plume_diagnostics(fields[label]),
                "logical_steps": work[label]["logical_steps"],
                "transport_solves": work[label]["transport_solves"],
                "reaction_evaluations": work[label]["reaction_evaluations"],
                "total_sia_iterations": work[label]["total_sia_iterations"],
                "wall_time_seconds": work[label]["wall_time_seconds"],
            }
        )

    archive = {
        f"{label}_{component}": fields[label][component]
        for label in fields
        for component in FIELD_NAMES
    }
    lens = reactive_lens_mask(NROW, NCOL)
    archive.update(
        {
            "hydraulic_conductivity_m_per_day": hydraulic_conductivity_field(
                NROW, NCOL
            ),
            "reactive_lens_mask": lens.astype(np.uint8),
            "solid_oxidant_capacity_model_mol": oxidant_capacity_field(
                NROW, NCOL
            ),
            "kinetic_rate_per_day": np.full(
                (NROW, NCOL), KINETIC_RATE_PER_DAY, dtype=float
            ),
            "x_cell_centers_m": (
                np.arange(NCOL, dtype=float) + 0.5
            )
            * LENGTH
            / NCOL,
            "y_cell_centers_m": (
                np.arange(NROW, dtype=float) + 0.5
            )
            * WIDTH
            / NROW,
            "domain_extent_m": np.array([0.0, LENGTH, 0.0, WIDTH]),
        }
    )
    np.savez(output_dir / "final_fields_comparison.npz", **archive)
    _write_metrics(rows, output_dir)

    row_lookup = {row["label"]: row for row in rows}
    coarse_errors = {
        method: row_lookup[method]["combined_nrmse"] for method in METHODS
    }
    transport_work = {
        method: row_lookup[method]["transport_solves"] for method in METHODS
    }
    wall_times = {
        method: row_lookup[method]["wall_time_seconds"] for method in METHODS
    }
    sia_diagnostics = work["SIA"]["sia_diagnostics"]
    validation = {
        "benchmark_claim": "SIA < Strang < SNIA error at reversed work cost",
        "primary_metric": "combined Don/solid-oxidant-extent NRMSE",
        "field_scales": FIELD_SCALES,
        "coarse_error_order": "SIA < Strang < SNIA",
        "coarse_combined_nrmse": coarse_errors,
        "deterministic_work_order": "SIA > Strang > SNIA",
        "coarse_transport_solves": transport_work,
        "observed_wall_time_seconds": wall_times,
        "observed_wall_time_has_expected_order": bool(
            wall_times["SIA"] > wall_times["Strang"] > wall_times["SNIA"]
        ),
        "reference_check_combined_nrmse": row_lookup["ReferenceCheck"][
            "combined_nrmse"
        ],
        "cross_method_reference_nrmse": row_lookup[
            "ReferenceCrossCheck"
        ]["combined_nrmse"],
        "reference_method": "Strang",
        "reference_step_days": 0.125,
        "reference_check_step_days": 0.25,
        "sia_all_steps_converged": bool(sia_diagnostics)
        and all(item["converged"] for item in sia_diagnostics),
        "sia_step_iterations": [
            int(item["iterations"]) for item in sia_diagnostics
        ],
        "scenario": {
            "pulse_duration_days": PULSE_DURATION,
            "flush_duration_days": FLUSH_DURATION,
            "coarse_steps_per_period": list(COARSE_STEPS),
            "matrix_oxidant_capacity_model_mol": MATRIX_OXIDANT_CAPACITY,
            "lens_oxidant_capacity_model_mol": LENS_OXIDANT_CAPACITY,
            "kinetic_rate_per_day": KINETIC_RATE_PER_DAY,
        },
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nErrors against the refined Strang reference")
    for row in rows:
        print(
            f"  {row['label']:14s}: combined NRMSE={row['combined_nrmse']:.6e}, "
            f"Don mass={row['aqueous_donor_mol']:.4e} mol, "
            f"extent sum={row['summed_oxidant_consumption_model_mol']:.4e}"
        )
    if not (
        coarse_errors["SIA"]
        < coarse_errors["Strang"]
        < coarse_errors["SNIA"]
    ):
        raise AssertionError("Expected combined NRMSE order SIA < Strang < SNIA")
    if not (
        transport_work["SIA"]
        > transport_work["Strang"]
        > transport_work["SNIA"]
    ):
        raise AssertionError("Expected transport-work order SIA > Strang > SNIA")
    if not validation["sia_all_steps_converged"]:
        raise AssertionError("At least one strict SIA logical step did not converge")
    if validation["reference_check_combined_nrmse"] > 0.005:
        raise AssertionError(
            "The 0.25-day Strang reference check differs too much from the "
            "0.125-day reference"
        )
    if validation["cross_method_reference_nrmse"] > 0.005:
        raise AssertionError(
            "The 0.125-day SNIA cross-check differs too much from the "
            "0.125-day Strang reference"
        )
    print(
        "Two-dimensional splitting validation passed; execute plot.ipynb "
        f"to create figures from {output_dir}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--steps", type=int, nargs=2, metavar=("PULSE", "FLUSH"))
    parser.add_argument("--label")
    arguments = parser.parse_args()
    os.chdir(REPO_ROOT)
    if arguments.method:
        if arguments.steps is None or arguments.label is None:
            parser.error("--method requires --steps and --label")
        realization, metadata = run_realization(
            arguments.method, tuple(arguments.steps), arguments.label
        )
        _persist_child(arguments.label, realization, metadata)
    else:
        run_comparison()
