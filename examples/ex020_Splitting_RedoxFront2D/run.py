"""Run a redox-front realization or compare the supported splitting methods."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex020_Splitting_RedoxFront2D.modflow_model import build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import MF6PQC, BackendPaths, CellFields, OutputOptions, SIAOptions, SimulationConfig
from mf6pqc.utils import get_gwt_model_name

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
REPO_ROOT = CASE_DIR.parents[1]


@dataclass(frozen=True)
class Config:
    matrix_conductivity: float
    channel_conductivity: float
    low_conductivity: float
    donor_concentration_scale: float
    matrix_oxidant_capacity: float
    lens_oxidant_capacity: float
    nrow: int
    ncol: int
    length: float
    width: float
    porosity: float
    pulse_duration: float
    flush_duration: float
    coarse_steps: tuple[int, ...]
    reference_check_steps: tuple[int, ...]
    reference_steps: tuple[int, ...]
    methods: tuple[str, ...]
    report_components: tuple[str, ...]
    kinetic_rate_per_day: float
    sia_max_iterations: int
    sia_rtol: float
    sia_atol: float
    sia_source_relaxation: float

    @property
    def nxyz(self):
        return self.nrow * self.ncol

    @property
    def field_names(self):
        return (*self.report_components, "Extent")

    @property
    def field_scales(self):
        return {"Don": self.donor_concentration_scale, "Extent": self.lens_oxidant_capacity}


def hydraulic_conductivity_field(nrow: int, ncol: int, *, config: Config) -> np.ndarray:
    rows, columns = np.indices((nrow, ncol))
    channel_center = 0.5 * (nrow - 1) + 1.8 * np.sin(2.0 * np.pi * columns / max(ncol - 1, 1))
    distance = np.abs(rows - channel_center)
    field = np.full((nrow, ncol), config.matrix_conductivity, dtype=float)
    field[distance <= 1.25] = config.channel_conductivity
    field[(columns > ncol // 2) & (distance > 3.5)] = config.low_conductivity
    return field


def reactive_lens_mask(nrow: int, ncol: int) -> np.ndarray:
    rows, columns = np.indices((nrow, ncol))
    lens_center = 0.72 * (nrow - 1) - 0.34 * columns
    return (
        (columns >= int(0.28 * ncol))
        & (columns <= int(0.76 * ncol))
        & (np.abs(rows - lens_center) <= 1.35)
    )


def oxidant_capacity_field(nrow: int, ncol: int, *, config: Config) -> np.ndarray:
    return np.where(
        reactive_lens_mask(nrow, ncol), config.lens_oxidant_capacity, config.matrix_oxidant_capacity
    )


def _run_paths(label: str) -> tuple[Path, Path]:
    return (
        runtime_path(__file__, "simulation") / label,
        runtime_path(__file__, "output") / "runs" / label,
    )


def _kinetics_zones(*, config: Config) -> np.ndarray:
    return np.where(reactive_lens_mask(config.nrow, config.ncol).ravel(), 2, 1).astype(np.int32)


MATRIX_OXIDANT_CAPACITY = 0.0002
LENS_OXIDANT_CAPACITY = 0.0008
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
FIELD_SCALES = {"Don": 0.001, "Extent": LENS_OXIDANT_CAPACITY}
KINETIC_RATE_PER_DAY = 5.0
SIA_MAX_ITERATIONS = 250
SIA_RTOL = 1e-06
SIA_ATOL = 2e-08
SIA_SOURCE_RELAXATION = 0.85
MATRIX_CONDUCTIVITY = 0.45
CHANNEL_CONDUCTIVITY = 1.35
LOW_CONDUCTIVITY = 0.25
DONOR_CONCENTRATION_SCALE = 0.001
CASE_CONFIG = Config(
    matrix_oxidant_capacity=MATRIX_OXIDANT_CAPACITY,
    lens_oxidant_capacity=LENS_OXIDANT_CAPACITY,
    nrow=NROW,
    ncol=NCOL,
    length=LENGTH,
    width=WIDTH,
    porosity=POROSITY,
    pulse_duration=PULSE_DURATION,
    flush_duration=FLUSH_DURATION,
    coarse_steps=COARSE_STEPS,
    reference_check_steps=REFERENCE_CHECK_STEPS,
    reference_steps=REFERENCE_STEPS,
    methods=METHODS,
    report_components=REPORT_COMPONENTS,
    kinetic_rate_per_day=KINETIC_RATE_PER_DAY,
    sia_max_iterations=SIA_MAX_ITERATIONS,
    sia_rtol=SIA_RTOL,
    sia_atol=SIA_ATOL,
    sia_source_relaxation=SIA_SOURCE_RELAXATION,
    matrix_conductivity=MATRIX_CONDUCTIVITY,
    channel_conductivity=CHANNEL_CONDUCTIVITY,
    low_conductivity=LOW_CONDUCTIVITY,
    donor_concentration_scale=DONOR_CONCENTRATION_SCALE,
)
INLET_HEAD = 0.8
OUTLET_HEAD = 0.0
TOP = 1.0
BOTM = 0.0
ALH = 0.12
ATH1 = 0.012
DIFFC = 0.0


def run_realization(
    method: str, logical_steps: tuple[int, int], label: str
) -> tuple[dict[str, np.ndarray], dict]:
    """Run one splitting realization and return its final fields and diagnostics."""
    workspace, output_dir = _run_paths(label)
    simulation_config = SimulationConfig(
        case_name="ex020",
        nxyz=NXYZ,
        nthreads=2,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(temperature_c=20.0, porosity=POROSITY, saturation=1.0),
        sia=SIAOptions(
            maximum_iterations=SIA_MAX_ITERATIONS,
            relative_tolerance=SIA_RTOL,
            absolute_tolerance=SIA_ATOL,
            source_relaxation=SIA_SOURCE_RELAXATION,
            fail_on_nonconvergence=True,
        ),
        output=OutputOptions(progress_interval=100),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(
            ic_map={"solution": 0, "kinetics": _kinetics_zones(config=CASE_CONFIG)}
        )
        pulse_concentrations = simulator.get_initial_concentrations(1)
        background_concentrations = simulator.get_initial_concentrations(0)
        species = simulator.get_components()
        missing = sorted(set(REPORT_COMPONENTS) - set(species))
        if missing:
            raise RuntimeError(f"PHREEQC did not expose components: {missing}")
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            background_concentrations=background_concentrations,
            pulse_concentrations=pulse_concentrations,
            nrow=NROW,
            ncol=NCOL,
            length=LENGTH,
            width=WIDTH,
            top=TOP,
            botm=BOTM,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity_field(NROW, NCOL, config=CASE_CONFIG),
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            flush_duration=FLUSH_DURATION,
            logical_steps_per_period=logical_steps,
            pulse_duration=PULSE_DURATION,
            strang_half_steps=method == "Strang",
        )
        simulator.run(method=method)
        fields = {}
        for component in REPORT_COMPONENTS:
            model_name = get_gwt_model_name(component)
            address = simulator.modflow_api.get_var_address("X", model_name)
            fields[component] = np.asarray(
                simulator.modflow_api.get_value(address), dtype=float
            ).reshape(NROW, NCOL)
        heading_lookup = {
            heading.casefold(): index for index, heading in enumerate(simulator.headings)
        }
        try:
            extent_index = heading_lookup["redox_extent"]
        except KeyError as exc:
            raise RuntimeError("Selected output must contain the Redox_extent heading") from exc
        fields["Extent"] = np.asarray(simulator.selected_output[extent_index], dtype=float).reshape(
            NROW, NCOL
        )
        sia_iterations = int(np.sum(simulator.sia_iterations)) if method == "SIA" else None
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
        return (fields, metadata)


def _persist_child(label: str, fields: dict[str, np.ndarray], metadata: dict) -> None:
    _, output_dir = _run_paths(label)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(output_dir / "final_fields.npz", **fields)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _launch(
    method: str, logical_steps: tuple[int, int], label: str, *, config: Config
) -> tuple[dict[str, np.ndarray], dict]:
    command = [
        sys.executable,
        str(CASE_DIR / "run.py"),
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
        fields = {name: archive[name].copy() for name in config.field_names}
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    return (fields, metadata)


def _validate_fields(label: str, fields: dict[str, np.ndarray], *, config: Config) -> None:
    for component, field in fields.items():
        if field.shape != (config.nrow, config.ncol) or not np.all(np.isfinite(field)):
            raise AssertionError(f"{label}/{component} is not a finite 2-D field")
        if np.min(field) < -2e-10:
            raise AssertionError(f"{label}/{component} contains negative concentration")
    if np.max(fields["Don"]) > 0.001003:
        raise AssertionError(f"{label} donor exceeds the inlet maximum")
    capacity = oxidant_capacity_field(config.nrow, config.ncol, config=config)
    if np.any(fields["Extent"] > capacity + 5e-08):
        raise AssertionError(f"{label} exceeds the local solid-oxidant capacity")
    if np.min(fields["Extent"]) < -2e-10:
        raise AssertionError(f"{label} has an invalid cumulative reaction extent")


def field_error_metrics(
    fields: dict[str, np.ndarray], reference: dict[str, np.ndarray], *, config: Config
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    normalized_differences = []
    for field_name in config.field_names:
        difference = fields[field_name] - reference[field_name]
        scale = config.field_scales[field_name]
        metrics[f"{field_name}_rmse"] = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{field_name}_nrmse"] = float(np.sqrt(np.mean((difference / scale) ** 2)))
        metrics[f"{field_name}_linf"] = float(np.max(np.abs(difference)))
        normalized_differences.append((difference / scale).ravel())
    metrics["combined_nrmse"] = float(np.sqrt(np.mean(np.concatenate(normalized_differences) ** 2)))
    return metrics


def plume_diagnostics(fields: dict[str, np.ndarray], *, config: Config) -> dict[str, float]:
    delr = config.length / config.ncol
    delc = config.width / config.nrow
    water_volume_litres = delr * delc * config.porosity * 1000.0
    x = (np.arange(config.ncol) + 0.5) * delr
    donor = fields["Don"]
    extent = fields["Extent"]
    capacity = oxidant_capacity_field(config.nrow, config.ncol, config=config)
    donor_mass = float(np.sum(donor) * water_volume_litres)
    donor_by_column = np.sum(donor, axis=0)
    centroid = (
        float(np.sum(donor_by_column * x) / np.sum(donor_by_column))
        if np.sum(donor_by_column) > 0.0
        else 0.0
    )
    lens = reactive_lens_mask(config.nrow, config.ncol)
    lens_extent_fraction = (
        float(np.sum(extent[lens]) / np.sum(extent)) if np.sum(extent) > 0.0 else 0.0
    )
    active_area = float(np.count_nonzero((donor > 5e-05) & (extent > 1e-06)) * delr * delc)
    depleted_area = float(np.count_nonzero(extent / capacity >= 0.9) * delr * delc)
    return {
        "aqueous_donor_mol": donor_mass,
        "summed_oxidant_consumption_model_mol": float(np.sum(extent)),
        "oxidant_capacity_utilization_fraction": float(np.sum(extent) / np.sum(capacity)),
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
    with (output_dir / "comparison_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_comparison(*, config: Config) -> None:
    output_dir = runtime_path(__file__, "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    fields: dict[str, dict[str, np.ndarray]] = {}
    work: dict[str, dict] = {}
    for method in config.methods:
        fields[method], work[method] = _launch(
            method, config.coarse_steps, f"coarse_{method.lower()}", config=config
        )
        _validate_fields(method, fields[method], config=config)
    fields["ReferenceCheck"], work["ReferenceCheck"] = _launch(
        "Strang", config.reference_check_steps, "reference_check_strang", config=config
    )
    _validate_fields("ReferenceCheck", fields["ReferenceCheck"], config=config)
    fields["Reference"], work["Reference"] = _launch(
        "Strang", config.reference_steps, "reference_strang", config=config
    )
    _validate_fields("Reference", fields["Reference"], config=config)
    fields["ReferenceCrossCheck"], work["ReferenceCrossCheck"] = _launch(
        "SNIA", config.reference_steps, "reference_crosscheck_snia", config=config
    )
    _validate_fields("ReferenceCrossCheck", fields["ReferenceCrossCheck"], config=config)
    rows = []
    for label in (*config.methods, "ReferenceCheck", "ReferenceCrossCheck", "Reference"):
        errors = (
            field_error_metrics(fields[label], fields["Reference"], config=config)
            if label != "Reference"
            else {
                f"{component}_{metric}": 0.0
                for component in config.field_names
                for metric in ("rmse", "nrmse", "linf")
            }
        )
        errors["combined_nrmse"] = errors.get("combined_nrmse", 0.0)
        rows.append(
            {
                "label": label,
                "method": work[label]["method"],
                **errors,
                **plume_diagnostics(fields[label], config=config),
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
        for component in config.field_names
    }
    lens = reactive_lens_mask(config.nrow, config.ncol)
    archive.update(
        {
            "hydraulic_conductivity_m_per_day": hydraulic_conductivity_field(
                config.nrow, config.ncol, config=config
            ),
            "reactive_lens_mask": lens.astype(np.uint8),
            "solid_oxidant_capacity_model_mol": oxidant_capacity_field(
                config.nrow, config.ncol, config=config
            ),
            "kinetic_rate_per_day": np.full(
                (config.nrow, config.ncol), config.kinetic_rate_per_day, dtype=float
            ),
            "x_cell_centers_m": (np.arange(config.ncol, dtype=float) + 0.5)
            * config.length
            / config.ncol,
            "y_cell_centers_m": (np.arange(config.nrow, dtype=float) + 0.5)
            * config.width
            / config.nrow,
            "domain_extent_m": np.array([0.0, config.length, 0.0, config.width]),
        }
    )
    np.savez(output_dir / "final_fields_comparison.npz", **archive)
    _write_metrics(rows, output_dir)
    row_lookup = {row["label"]: row for row in rows}
    coarse_errors = {method: row_lookup[method]["combined_nrmse"] for method in config.methods}
    transport_work = {method: row_lookup[method]["transport_solves"] for method in config.methods}
    wall_times = {method: row_lookup[method]["wall_time_seconds"] for method in config.methods}
    sia_diagnostics = work["SIA"]["sia_diagnostics"]
    validation = {
        "benchmark_claim": "SIA < Strang < SNIA error at reversed work cost",
        "primary_metric": "combined Don/solid-oxidant-extent NRMSE",
        "field_scales": config.field_scales,
        "coarse_error_order": "SIA < Strang < SNIA",
        "coarse_combined_nrmse": coarse_errors,
        "deterministic_work_order": "SIA > Strang > SNIA",
        "coarse_transport_solves": transport_work,
        "observed_wall_time_seconds": wall_times,
        "observed_wall_time_has_expected_order": bool(
            wall_times["SIA"] > wall_times["Strang"] > wall_times["SNIA"]
        ),
        "reference_check_combined_nrmse": row_lookup["ReferenceCheck"]["combined_nrmse"],
        "cross_method_reference_nrmse": row_lookup["ReferenceCrossCheck"]["combined_nrmse"],
        "reference_method": "Strang",
        "reference_step_days": 0.125,
        "reference_check_step_days": 0.25,
        "sia_all_steps_converged": bool(sia_diagnostics)
        and all(item["converged"] for item in sia_diagnostics),
        "sia_step_iterations": [int(item["iterations"]) for item in sia_diagnostics],
        "scenario": {
            "pulse_duration_days": config.pulse_duration,
            "flush_duration_days": config.flush_duration,
            "coarse_steps_per_period": list(config.coarse_steps),
            "matrix_oxidant_capacity_model_mol": config.matrix_oxidant_capacity,
            "lens_oxidant_capacity_model_mol": config.lens_oxidant_capacity,
            "kinetic_rate_per_day": config.kinetic_rate_per_day,
        },
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\nErrors against the refined Strang reference")
    for row in rows:
        print(
            f"  {row['label']:14s}: combined NRMSE={row['combined_nrmse']:.6e}, Don mass={row['aqueous_donor_mol']:.4e} mol, extent sum={row['summed_oxidant_consumption_model_mol']:.4e}"
        )
    if not coarse_errors["SIA"] < coarse_errors["Strang"] < coarse_errors["SNIA"]:
        raise AssertionError("Expected combined NRMSE order SIA < Strang < SNIA")
    if not transport_work["SIA"] > transport_work["Strang"] > transport_work["SNIA"]:
        raise AssertionError("Expected transport-work order SIA > Strang > SNIA")
    if not validation["sia_all_steps_converged"]:
        raise AssertionError("At least one strict SIA logical step did not converge")
    if validation["reference_check_combined_nrmse"] > 0.005:
        raise AssertionError(
            "The 0.25-day Strang reference check differs too much from the 0.125-day reference"
        )
    if validation["cross_method_reference_nrmse"] > 0.005:
        raise AssertionError(
            "The 0.125-day SNIA cross-check differs too much from the 0.125-day Strang reference"
        )
    print(
        f"Two-dimensional splitting validation passed; execute plot.ipynb to create figures from {output_dir}"
    )


def main() -> None:
    """Parse the command line and run one realization or the full comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--steps", type=int, nargs=2, metavar=("PULSE", "FLUSH"))
    parser.add_argument("--label")
    args = parser.parse_args()
    if args.method:
        if args.steps is None or args.label is None:
            parser.error("--method requires --steps and --label")
        realization, metadata = run_realization(args.method, tuple(args.steps), args.label)
        _persist_child(args.label, realization, metadata)
    else:
        run_comparison(config=CASE_CONFIG)


if __name__ == "__main__":
    configure_logging()
    main()
