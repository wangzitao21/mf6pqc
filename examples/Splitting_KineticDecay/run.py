from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc


EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from mf6pqc import MF6PQC
from mf6pqc.utils import get_gwt_model_name
from modflow_model import build_transport_model


DAYS_PER_YEAR = 365.25
NGRID_NODES = 16
NINTERIOR_NODES = NGRID_NODES - 1
LENGTH_M = 6.0
GRID_SPACING_M = 0.4
VOLUMETRIC_WATER_CONTENT = 1.0
VELOCITY_M_PER_YEAR = 100.0
DISPERSIVITY_M = 0.2
FINAL_TIME_YEARS = 0.5
DECAY_RATE_PER_YEAR = 100.0
VELOCITY_M_PER_DAY = VELOCITY_M_PER_YEAR / DAYS_PER_YEAR
FINAL_TIME_DAYS = FINAL_TIME_YEARS * DAYS_PER_YEAR
DECAY_RATE_PER_DAY = DECAY_RATE_PER_YEAR / DAYS_PER_YEAR
PAPER_CFL_VALUES = (0.1, 0.5, 1.0)
METHODS = ("SNIA", "Strang", "SIA")


class PhreeqcInstantaneousRateEvaluator:

    def __init__(self) -> None:
        self.simulator: MF6PQC | None = None

    def bind(self, simulator: MF6PQC) -> None:
        self.simulator = simulator

    def __call__(
        self,
        components: tuple[str, ...],
        concentrations: np.ndarray,
        target_time_days: float,
    ) -> np.ndarray:
        simulator = self.simulator
        if simulator is None:
            raise RuntimeError("The PhreeqcRM rate evaluator is not bound")
        state_id = 3
        simulator.phreeqc_rm.StateSave(state_id)
        try:
            simulator.phreeqc_rm.SetConcentrations(
                np.asarray(concentrations, dtype=float).reshape(-1)
            )
            simulator.phreeqc_rm.SetTime(
                target_time_days * 24.0 * 60.0 * 60.0
            )
            simulator.phreeqc_rm.SetTimeStep(0.0)
            simulator.phreeqc_rm.RunCells()
            selected = np.asarray(
                simulator.phreeqc_rm.GetSelectedOutput(), dtype=float
            ).reshape(-1, simulator.nxyz)
        finally:
            simulator.phreeqc_rm.StateApply(state_id)
            simulator.phreeqc_rm.StateDelete(state_id)

        rates = np.zeros_like(concentrations)
        spe_index = components.index("Spe")
        rate_index = simulator.headings.index("Spe_rate_per_day")
        rates[spe_index] = selected[rate_index]
        # Node 0 is the prescribed, nonreactive Dirichlet boundary.
        rates[spe_index, 0] = 0.0
        return rates


def analytical_solution(x_m: np.ndarray) -> np.ndarray:
    """Return Bear's semi-infinite advection-dispersion-decay solution."""
    x_m = np.asarray(x_m, dtype=float)
    dispersion = VELOCITY_M_PER_YEAR * DISPERSIVITY_M
    root = np.sqrt(
        VELOCITY_M_PER_YEAR**2
        + 4.0 * DECAY_RATE_PER_YEAR * dispersion
    )
    scale = 2.0 * np.sqrt(dispersion * FINAL_TIME_YEARS)
    return 0.5 * np.exp(VELOCITY_M_PER_YEAR * x_m / (2.0 * dispersion)) * (
        np.exp(-root * x_m / (2.0 * dispersion))
        * erfc((x_m - root * FINAL_TIME_YEARS) / scale)
        + np.exp(root * x_m / (2.0 * dispersion))
        * erfc((x_m + root * FINAL_TIME_YEARS) / scale)
    )


def logical_steps_for_cfl(cfl: float) -> int:
    """Map the paper CFL number to an integer logical-step count."""
    if cfl <= 0.0:
        raise ValueError("CFL must be positive")
    raw_steps = (
        VELOCITY_M_PER_YEAR
        * FINAL_TIME_YEARS
        / (GRID_SPACING_M * cfl)
    )
    steps = int(round(raw_steps))
    if not np.isclose(raw_steps, steps, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"CFL={cfl:g} does not divide the fixed final time into an "
            "integer number of logical steps"
        )
    return steps


def _cfl_token(cfl: float) -> str:
    return format(cfl, "g").replace(".", "p")


def _profile_key(method: str, cfl: float) -> str:
    return f"profile__{method}__cfl_{_cfl_token(cfl)}"


def _run_directory(base: str, method: str, cfl: float) -> Path:
    return (
        EXAMPLE_DIR
        / base
        / "paper_figure6"
        / f"cfl_{_cfl_token(cfl)}"
        / method.lower()
    )


def run_method(method: str, cfl: float) -> tuple[np.ndarray, dict]:
    """Run one process-isolated realization of the paper problem."""
    logical_steps = logical_steps_for_cfl(cfl)
    workspace = _run_directory("simulation", method, cfl)
    output_dir = _run_directory("output", method, cfl)
    rate_evaluator = (
        PhreeqcInstantaneousRateEvaluator() if method == "SIA" else None
    )
    simulator = MF6PQC(
        case_name=(
            f"steefel1996_decay_{method.lower()}_cfl_{_cfl_token(cfl)}"
        ),
        nxyz=NGRID_NODES,
        nthreads=2,
        porosity=VOLUMETRIC_WATER_CONTENT,
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
        progress_interval=250,
        sia_max_iterations=80,
        sia_rtol=1.0e-6,
        sia_atol=1.0e-9,
        sia_source_relaxation=1.0,
        sia_fail_on_nonconvergence=True,
        sia_rate_evaluator=rate_evaluator,
    )
    if rate_evaluator is not None:
        rate_evaluator.bind(simulator)
    try:
        # SNIA/Strang advance KINETICS 1 in the interior.  Paper-rate SIA
        # applies decay through GWT-SRC, so all of its KINETICS zones are
        # deliberately inert and no hidden second reaction can occur.
        kinetic_zones = np.ones(NGRID_NODES, dtype=np.int32)
        if method == "SIA":
            kinetic_zones.fill(0)
        else:
            kinetic_zones[0] = 0
        initial = simulator.setup(
            {"solution": 0, "kinetics": kinetic_zones}
        )
        boundary = simulator.get_initial_concentrations(1)
        build_transport_model(
            str(workspace),
            simulator.get_components(),
            initial,
            boundary,
            ncol=NGRID_NODES,
            length=LENGTH_M,
            porosity=VOLUMETRIC_WATER_CONTENT,
            pore_velocity=VELOCITY_M_PER_DAY,
            dispersivity=DISPERSIVITY_M,
            perlen=FINAL_TIME_DAYS,
            nstp=(2 * logical_steps if method == "Strang" else logical_steps),
            boundary_node_species="Spe",
            # Pe=2 is within the stable range of the central scheme and this
            # is closer to the paper's control-volume finite-element stencil
            # than MODFLOW's nonlinear TVD limiter.
            advection_scheme="CENTRAL",
        )
        getattr(simulator, f"run_{method}")()
        address = simulator.modflow_api.get_var_address(
            "X", get_gwt_model_name("Spe")
        )
        full_profile = np.asarray(
            simulator.modflow_api.get_value(address), dtype=float
        ).copy()
        spe_index = simulator.get_components().index("Spe")
        inlet_concentration = float(boundary[spe_index])
        # Node 0 represents the prescribed x=0 boundary.  Figure 6 reports
        # the 15 downstream nodes x=0.4,...,6.0 m.
        # Normalize by PhreeqcRM's mol/L value so the analytical boundary is
        # exactly C/C0=1 despite PHREEQC solution-volume conversion.
        profile = full_profile[1:].copy() / inlet_concentration
        total_sia_iterations = (
            int(np.sum(simulator.sia_iterations)) if method == "SIA" else None
        )
        logical_dt_days = FINAL_TIME_DAYS / logical_steps
        metadata = {
            "method": method,
            "cfl": float(
                VELOCITY_M_PER_DAY
                * logical_dt_days
                / GRID_SPACING_M
            ),
            "logical_steps": logical_steps,
            "logical_dt_days": logical_dt_days,
            "logical_dt_years": logical_dt_days / DAYS_PER_YEAR,
            "damkohler_per_step": DECAY_RATE_PER_DAY * logical_dt_days,
            "phreeqc_inlet_concentration_mol_per_l": inlet_concentration,
            "boundary_endpoint_concentration": float(
                full_profile[0] / inlet_concentration
            ),
            "advection_scheme": "CENTRAL",
            "sia_source_form": (
                "instantaneous PhreeqcRM USER_PUNCH rate, paper equation (108)"
                if method == "SIA"
                else None
            ),
            "reported_interior_nodes": NINTERIOR_NODES,
            "transport_solves": {
                "SNIA": logical_steps,
                "Strang": 2 * logical_steps,
                "SIA": total_sia_iterations,
            }[method],
            "reaction_evaluations": {
                "SNIA": logical_steps,
                "Strang": logical_steps,
                "SIA": total_sia_iterations,
            }[method],
            "full_phreeqc_reaction_steps": (
                logical_steps if method in {"SNIA", "Strang"} else 0
            ),
            "instantaneous_phreeqc_rate_evaluations": (
                total_sia_iterations if method == "SIA" else 0
            ),
            "diagnostic_respeciations": (
                logical_steps if method == "Strang" else 0
            ),
            "total_sia_iterations": total_sia_iterations,
            "wall_time_seconds": simulator.last_run_wall_time_seconds,
            "sia_diagnostics": list(simulator.sia_diagnostics),
        }
        return profile, metadata
    finally:
        simulator.finalize()


def _persist_child_result(
    method: str, cfl: float, profile: np.ndarray, metadata: dict
) -> None:
    output_dir = _run_directory("output", method, cfl)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "final_profile.npy", profile)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _launch_method(method: str, cfl: float) -> tuple[np.ndarray, dict]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--method",
        method,
        "--cfl",
        format(cfl, "g"),
    ]
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    output_dir = _run_directory("output", method, cfl)
    profile = np.load(output_dir / "final_profile.npy")
    metadata = json.loads(
        (output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    return profile, metadata


def error_metrics(profile: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return the paper L2 norm plus scale-independent profile norms."""
    difference = np.asarray(profile) - np.asarray(reference)
    return {
        "l2": float(np.linalg.norm(difference)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "l1_mean": float(np.mean(np.abs(difference))),
        "linf": float(np.max(np.abs(difference))),
    }


def _validate_profile(method: str, cfl: float, profile: np.ndarray) -> None:
    if profile.shape != (NINTERIOR_NODES,) or not np.all(np.isfinite(profile)):
        raise AssertionError(
            f"{method}/CFL={cfl:g} produced an invalid final profile"
        )
    if np.min(profile) < -1.0e-12 or np.max(profile) > 1.0 + 1.0e-8:
        raise AssertionError(
            f"{method}/CFL={cfl:g} violates the [0, 1] concentration range"
        )


def _realizations() -> list[tuple[str, float]]:
    return [
        (method, cfl)
        for method in METHODS
        for cfl in reversed(PAPER_CFL_VALUES)
    ]


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_paper_replication() -> None:
    """Run the Figure 6 matrix and validate the algorithm-level signatures."""
    x_paper_nodes = (
        np.arange(1, NGRID_NODES, dtype=float) * GRID_SPACING_M
    )
    x_cell_centers = (
        np.arange(1, NGRID_NODES, dtype=float) + 0.5
    ) * GRID_SPACING_M
    x_analytical_dense = np.linspace(0.0, LENGTH_M, 601)
    analytical_paper_nodes = analytical_solution(x_paper_nodes)
    analytical_cell_centers = analytical_solution(x_cell_centers)
    analytical_dense = analytical_solution(x_analytical_dense)
    profiles: dict[tuple[str, float], np.ndarray] = {}
    metadata: dict[tuple[str, float], dict] = {}
    rows: list[dict] = []

    for method, cfl in _realizations():
        profile, run_metadata = _launch_method(method, cfl)
        _validate_profile(method, cfl, profile)
        profiles[(method, cfl)] = profile
        metadata[(method, cfl)] = run_metadata
        paper_errors = error_metrics(profile, analytical_paper_nodes)
        center_errors = error_metrics(profile, analytical_cell_centers)
        rows.append(
            {
                "method": method,
                "cfl": cfl,
                "logical_steps": run_metadata["logical_steps"],
                "logical_dt_years": run_metadata["logical_dt_years"],
                "damkohler_per_step": run_metadata["damkohler_per_step"],
                "boundary_endpoint_concentration": run_metadata[
                    "boundary_endpoint_concentration"
                ],
                "paper_node_l2": paper_errors["l2"],
                "paper_node_rmse": paper_errors["rmse"],
                "paper_node_l1_mean": paper_errors["l1_mean"],
                "paper_node_linf": paper_errors["linf"],
                "cell_center_l2": center_errors["l2"],
                "cell_center_rmse": center_errors["rmse"],
                "cell_center_l1_mean": center_errors["l1_mean"],
                "cell_center_linf": center_errors["linf"],
                "transport_solves": run_metadata["transport_solves"],
                "reaction_evaluations": run_metadata["reaction_evaluations"],
                "full_phreeqc_reaction_steps": run_metadata[
                    "full_phreeqc_reaction_steps"
                ],
                "instantaneous_phreeqc_rate_evaluations": run_metadata[
                    "instantaneous_phreeqc_rate_evaluations"
                ],
                "total_sia_iterations": run_metadata["total_sia_iterations"],
                "wall_time_seconds": run_metadata["wall_time_seconds"],
            }
        )

    output_dir = EXAMPLE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_arrays = {
        _profile_key(method, cfl): profile
        for (method, cfl), profile in profiles.items()
    }
    np.savez(
        output_dir / "paper_figure6_data.npz",
        x_paper_nodes_m=x_paper_nodes,
        x_modflow_cell_centers_m=x_cell_centers,
        x_analytical_dense_m=x_analytical_dense,
        analytical_paper_nodes=analytical_paper_nodes,
        analytical_cell_centers=analytical_cell_centers,
        analytical_dense=analytical_dense,
        **archive_arrays,
    )
    _write_csv(rows, output_dir / "paper_figure6_metrics.csv")
    (output_dir / "paper_figure6_metrics.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    row_lookup = {(row["method"], row["cfl"]): row for row in rows}
    cfl_one_errors = {
        method: row_lookup[(method, 1.0)]["paper_node_rmse"]
        for method in METHODS
    }
    same_cfl_order = {
        format(cfl, "g"): {
            method: row_lookup[(method, cfl)]["paper_node_rmse"]
            for method in METHODS
        }
        for cfl in PAPER_CFL_VALUES
    }
    sia_diagnostics = [
        item
        for cfl in PAPER_CFL_VALUES
        for item in metadata[("SIA", cfl)]["sia_diagnostics"]
    ]
    validation = {
        "source_case": (
            "Steefel and MacQuarrie (1996), equation (114) and Figure 6"
        ),
        "reference_type": "analytical semi-infinite solution",
        "sia_formulation": (
            "instantaneous endpoint reaction rate R^(n+1,m), as written in "
            "Steefel and MacQuarrie (1996), equation (108)"
        ),
        "noniterative_reaction_formulation": (
            "SNIA and Strang use the full PhreeqcRM kinetic reaction map"
        ),
        "paper_parameters": {
            "volumetric_water_content": VOLUMETRIC_WATER_CONTENT,
            "velocity_m_per_year": VELOCITY_M_PER_YEAR,
            "dispersivity_m": DISPERSIVITY_M,
            "grid_spacing_m": GRID_SPACING_M,
            "decay_rate_per_year": DECAY_RATE_PER_YEAR,
            "final_time_years": FINAL_TIME_YEARS,
            "grid_peclet": GRID_SPACING_M / DISPERSIVITY_M,
            "advection_scheme": "CENTRAL",
            "cfl_values": list(PAPER_CFL_VALUES),
        },
        "coordinate_conventions": {
            "paper_figure": (
                "explicit CNC boundary node x=0 followed by normalized "
                "interior concentrations at x=j*dx, j=1..15"
            ),
            "modflow_finite_volume": (
                "raw DIS centers of the reported cells x=(j+0.5)*dx, "
                "j=1..15; retained only as a grid-coordinate sensitivity"
            ),
        },
        "cfl_1_paper_node_rmse": cfl_one_errors,
        "same_cfl_paper_node_rmse": same_cfl_order,
        "asserted_order_at_cfl_1": "SIA < Strang < SNIA",
        "asserted_work_order_at_cfl_1": "SNIA < Strang < SIA",
        "sia_accuracy_threshold_rmse": 5.0e-3,
        "snia_requires_refinement": (
            "SNIA at CFL=0.1 must be more accurate than SNIA at CFL=1"
        ),
        "sia_all_steps_converged": bool(sia_diagnostics)
        and all(item["converged"] for item in sia_diagnostics),
        "sia_total_iterations": {
            format(cfl, "g"): metadata[("SIA", cfl)][
                "total_sia_iterations"
            ]
            for cfl in PAPER_CFL_VALUES
        },
        "boundary_condition_preserved": all(
            row["boundary_endpoint_concentration"] > 0.999 for row in rows
        ),
        "runs": rows,
    }
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("\nErrors against the analytical solution (paper-node convention)")
    for row in sorted(rows, key=lambda item: (item["method"], item["cfl"])):
        print(
            f"  {row['method']:7s} CFL={row['cfl']:3.1f}: "
            f"L2={row['paper_node_l2']:.6e}, "
            f"RMSE={row['paper_node_rmse']:.6e}, "
            f"cell-center RMSE={row['cell_center_rmse']:.6e}"
        )

    if not (
        cfl_one_errors["SIA"]
        < cfl_one_errors["Strang"]
        < cfl_one_errors["SNIA"]
    ):
        raise AssertionError(
            "Expected Figure 6 at CFL=1 to order paper-node RMSE as "
            "SIA < Strang < SNIA"
        )
    if cfl_one_errors["SIA"] > validation["sia_accuracy_threshold_rmse"]:
        raise AssertionError(
            "Paper-rate SIA did not stay within the 5e-3 RMSE acceptance "
            "threshold"
        )
    if not (
        row_lookup[("SNIA", 0.1)]["paper_node_rmse"]
        < row_lookup[("SNIA", 1.0)]["paper_node_rmse"]
    ):
        raise AssertionError("SNIA did not improve after reducing its time step")
    cfl_one_work = {
        method: row_lookup[(method, 1.0)]["transport_solves"]
        for method in METHODS
    }
    if not (
        cfl_one_work["SNIA"]
        < cfl_one_work["Strang"]
        < cfl_one_work["SIA"]
    ):
        raise AssertionError(
            "Expected CFL=1 transport work to order as SNIA < Strang < SIA"
        )
    if not validation["sia_all_steps_converged"]:
        raise AssertionError("At least one strict SIA logical step did not converge")
    if not validation["boundary_condition_preserved"]:
        raise AssertionError(
            "The explicit x=0 Dirichlet node was altered by a coupling substep"
        )
    if profiles[("SNIA", 1.0)][0] >= analytical_paper_nodes[0]:
        raise AssertionError(
            "SNIA did not reproduce the paper's inlet over-reaction signature"
        )
    print(
        "Paper-rate SIA validation passed; execute plot.ipynb or "
        "plot_results.py to create figures."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--cfl", type=float, default=1.0)
    arguments = parser.parse_args()
    os.chdir(REPO_ROOT)
    if arguments.method:
        result, run_metadata = run_method(arguments.method, arguments.cfl)
        _persist_child_result(
            arguments.method, arguments.cfl, result, run_metadata
        )
    else:
        run_paper_replication()
