"""Run the two-dimensional redox-front comparison."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np
import pandas as pd

from examples.ex020_Splitting_RedoxFront2D.modflow_model import build_model
from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    OutputOptions,
    SIAOptions,
    SimulationConfig,
)
from mf6pqc.utils import get_gwt_model_name

INPUT_DIR = CASE_DIR / "input_data"
WORKSPACE = CASE_DIR / "simulation"
OUTPUT_DIR = CASE_DIR / "output"
MODFLOW_LIBRARY = Path(
    os.environ.get(
        "MF6PQC_LIBMF6",
        CASE_DIR.parents[1]
        / "bin"
        / "mf6.8.0"
        / {"win32": "libmf6.dll", "darwin": "libmf6.dylib"}.get(sys.platform, "libmf6.so"),
    )
)


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
INLET_HEAD = 0.8
OUTLET_HEAD = 0.0
TOP = 1.0
BOTM = 0.0
ALH = 0.12
ATH1 = 0.012
DIFFC = 0.0


def hydraulic_conductivity_field(nrow: int, ncol: int) -> np.ndarray:
    rows, columns = np.indices((nrow, ncol))
    channel_center = 0.5 * (nrow - 1) + 1.8 * np.sin(2.0 * np.pi * columns / max(ncol - 1, 1))
    distance = np.abs(rows - channel_center)
    field = np.full((nrow, ncol), MATRIX_CONDUCTIVITY, dtype=float)
    field[distance <= 1.25] = CHANNEL_CONDUCTIVITY
    field[(columns > ncol // 2) & (distance > 3.5)] = LOW_CONDUCTIVITY
    return field


def reactive_lens_mask(nrow: int, ncol: int) -> np.ndarray:
    rows, columns = np.indices((nrow, ncol))
    lens_center = 0.72 * (nrow - 1) - 0.34 * columns
    return (
        (columns >= int(0.28 * ncol))
        & (columns <= int(0.76 * ncol))
        & (np.abs(rows - lens_center) <= 1.35)
    )


def oxidant_capacity_field(nrow: int, ncol: int) -> np.ndarray:
    return np.where(reactive_lens_mask(nrow, ncol), LENS_OXIDANT_CAPACITY, MATRIX_OXIDANT_CAPACITY)


def _run_paths(label: str) -> tuple[Path, Path]:
    return (WORKSPACE / label, OUTPUT_DIR / "runs" / label)


def _kinetics_zones() -> np.ndarray:
    return np.where(reactive_lens_mask(NROW, NCOL).ravel(), 2, 1).astype(np.int32)


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
            modflow_library=MODFLOW_LIBRARY,
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(temperature_c=20.0, porosity=POROSITY),
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
            ic_map={"solution": 0, "kinetics": _kinetics_zones()}
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
            hydraulic_conductivity=hydraulic_conductivity_field(NROW, NCOL),
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
        simulator.save_results()
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
        }
        return (fields, metadata)


def field_error_metrics(
    fields: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    normalized_differences = []
    for field_name in FIELD_NAMES:
        difference = fields[field_name] - reference[field_name]
        scale = FIELD_SCALES[field_name]
        metrics[f"{field_name}_rmse"] = float(np.sqrt(np.mean(difference**2)))
        metrics[f"{field_name}_nrmse"] = float(np.sqrt(np.mean((difference / scale) ** 2)))
        metrics[f"{field_name}_linf"] = float(np.max(np.abs(difference)))
        normalized_differences.append((difference / scale).ravel())
    metrics["combined_nrmse"] = float(np.sqrt(np.mean(np.concatenate(normalized_differences) ** 2)))
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields: dict[str, dict[str, np.ndarray]] = {}
    work: dict[str, dict] = {}
    for method in METHODS:
        fields[method], work[method] = run_realization(
            method, COARSE_STEPS, f"coarse_{method.lower()}"
        )
    fields["ReferenceCheck"], work["ReferenceCheck"] = run_realization(
        "Strang", REFERENCE_CHECK_STEPS, "reference_check_strang"
    )
    fields["Reference"], work["Reference"] = run_realization(
        "Strang", REFERENCE_STEPS, "reference_strang"
    )
    fields["ReferenceCrossCheck"], work["ReferenceCrossCheck"] = run_realization(
        "SNIA", REFERENCE_STEPS, "reference_crosscheck_snia"
    )
    rows = []
    for label in (*METHODS, "ReferenceCheck", "ReferenceCrossCheck", "Reference"):
        errors = (
            field_error_metrics(fields[label], fields["Reference"])
            if label != "Reference"
            else {
                f"{component}_{metric}": 0.0
                for component in FIELD_NAMES
                for metric in ("rmse", "nrmse", "linf")
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
            "hydraulic_conductivity_m_per_day": hydraulic_conductivity_field(NROW, NCOL),
            "reactive_lens_mask": lens.astype(np.uint8),
            "solid_oxidant_capacity_model_mol": oxidant_capacity_field(NROW, NCOL),
            "kinetic_rate_per_day": np.full((NROW, NCOL), KINETIC_RATE_PER_DAY, dtype=float),
            "x_cell_centers_m": (np.arange(NCOL, dtype=float) + 0.5) * LENGTH / NCOL,
            "y_cell_centers_m": (np.arange(NROW, dtype=float) + 0.5) * WIDTH / NROW,
            "domain_extent_m": np.array([0.0, LENGTH, 0.0, WIDTH]),
        }
    )
    np.savez(OUTPUT_DIR / "final_fields_comparison.npz", **archive)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
