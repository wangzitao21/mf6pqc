"""Run the one-dimensional kinetic-decay comparison."""

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np
import pandas as pd
from scipy.special import erfc

from examples.ex019_Splitting_KineticDecay1D.modflow_model import build_model
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


DAYS_PER_YEAR = 365.25
NGRID_NODES = 16
NINTERIOR_NODES = NGRID_NODES - 1
LENGTH_M = 6.0
GRID_SPACING_M = LENGTH_M / NINTERIOR_NODES
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


def logical_steps_for_cfl(cfl: float) -> int:
    if cfl <= 0.0:
        raise ValueError("CFL must be positive")
    raw_steps = VELOCITY_M_PER_YEAR * FINAL_TIME_YEARS / (GRID_SPACING_M * cfl)
    steps = int(round(raw_steps))
    if not np.isclose(raw_steps, steps, rtol=0.0, atol=1e-10):
        raise ValueError(
            f"CFL={cfl:g} does not divide the fixed final time into an integer number of logical steps"
        )
    return steps


def _cfl_token(cfl: float) -> str:
    return format(cfl, "g").replace(".", "p")


def _profile_key(method: str, cfl: float) -> str:
    return f"profile__{method}__cfl_{_cfl_token(cfl)}"


def _run_directory(base: str, method: str, cfl: float) -> Path:
    return CASE_DIR / base / "paper_figure6" / f"cfl_{_cfl_token(cfl)}" / method.lower()


class PhreeqcInstantaneousRateEvaluator:
    """Evaluate instantaneous rates without retaining trial chemistry states."""

    def __init__(self) -> None:
        self.simulator: MF6PQC | None = None

    def bind(self, simulator: MF6PQC) -> None:
        self.simulator = simulator

    def __call__(
        self, components: tuple[str, ...], concentrations: np.ndarray, target_time_days: float
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
            simulator.phreeqc_rm.SetTime(target_time_days * 24.0 * 60.0 * 60.0)
            simulator.phreeqc_rm.SetTimeStep(0.0)
            simulator.phreeqc_rm.RunCells()
            selected = np.asarray(simulator.phreeqc_rm.GetSelectedOutput(), dtype=float).reshape(
                -1, simulator.nxyz
            )
        finally:
            simulator.phreeqc_rm.StateApply(state_id)
            simulator.phreeqc_rm.StateDelete(state_id)
        rates = np.zeros_like(concentrations)
        spe_index = components.index("Spe")
        rate_index = simulator.headings.index("Spe_rate_per_day")
        rates[spe_index] = selected[rate_index]
        rates[spe_index, 0] = 0.0
        return rates


def run_method(method: str, cfl: float) -> tuple[np.ndarray, dict]:
    """Run a splitting method at the requested CFL and return its final profile."""
    logical_steps = logical_steps_for_cfl(cfl)
    workspace = _run_directory("simulation", method, cfl)
    output_dir = _run_directory("output", method, cfl)
    rate_evaluator = PhreeqcInstantaneousRateEvaluator() if method == "SIA" else None
    simulation_config = SimulationConfig(
        case_name="ex019",
        nxyz=NGRID_NODES,
        nthreads=2,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(porosity=VOLUMETRIC_WATER_CONTENT),
        sia=SIAOptions(
            maximum_iterations=80,
            relative_tolerance=1e-06,
            absolute_tolerance=1e-09,
            source_relaxation=1.0,
            fail_on_nonconvergence=True,
            rate_evaluator=rate_evaluator,
        ),
        output=OutputOptions(progress_interval=250),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        if rate_evaluator is not None:
            rate_evaluator.bind(simulator)
        kinetic_zones = np.ones(NGRID_NODES, dtype=np.int32)
        if method == "SIA":
            kinetic_zones.fill(0)
        else:
            kinetic_zones[0] = 0
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "kinetics": kinetic_zones})
        inflow_concentrations = simulator.get_initial_concentrations(1)
        species = simulator.get_components()
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            ncol=NGRID_NODES,
            length=LENGTH_M,
            perlen=FINAL_TIME_DAYS,
            nstp=2 * logical_steps if method == "Strang" else logical_steps,
            porosity=VOLUMETRIC_WATER_CONTENT,
            alh=DISPERSIVITY_M,
            advection_scheme="CENTRAL",
            boundary_node_species="Spe",
            pore_velocity=VELOCITY_M_PER_DAY,
        )
        simulator.run(method=method)
        simulator.save_results()
        address = simulator.modflow_api.get_var_address("X", get_gwt_model_name("Spe"))
        full_profile = np.asarray(simulator.modflow_api.get_value(address), dtype=float).copy()
        spe_index = species.index("Spe")
        inlet_concentration = float(inflow_concentrations[spe_index])
        profile = full_profile[1:].copy() / inlet_concentration
        total_sia_iterations = int(np.sum(simulator.sia_iterations)) if method == "SIA" else None
        logical_dt_days = FINAL_TIME_DAYS / logical_steps
        metadata = {
            "method": method,
            "cfl": float(VELOCITY_M_PER_DAY * logical_dt_days / GRID_SPACING_M),
            "logical_steps": logical_steps,
            "logical_dt_days": logical_dt_days,
            "logical_dt_years": logical_dt_days / DAYS_PER_YEAR,
            "damkohler_per_step": DECAY_RATE_PER_DAY * logical_dt_days,
            "phreeqc_inlet_concentration_mol_per_l": inlet_concentration,
            "boundary_endpoint_concentration": float(full_profile[0] / inlet_concentration),
            "advection_scheme": "CENTRAL",
            "sia_source_form": "instantaneous PhreeqcRM USER_PUNCH rate, paper equation (108)"
            if method == "SIA"
            else None,
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
            "full_phreeqc_reaction_steps": logical_steps if method in {"SNIA", "Strang"} else 0,
            "instantaneous_phreeqc_rate_evaluations": total_sia_iterations
            if method == "SIA"
            else 0,
            "diagnostic_respeciations": logical_steps if method == "Strang" else 0,
            "total_sia_iterations": total_sia_iterations,
            "wall_time_seconds": simulator.last_run_wall_time_seconds,
        }
        return (profile, metadata)


def analytical_solution(x_m: np.ndarray) -> np.ndarray:
    x_m = np.asarray(x_m, dtype=float)
    dispersion = VELOCITY_M_PER_YEAR * DISPERSIVITY_M
    root = np.sqrt(VELOCITY_M_PER_YEAR**2 + 4.0 * DECAY_RATE_PER_YEAR * dispersion)
    scale = 2.0 * np.sqrt(dispersion * FINAL_TIME_YEARS)
    return (
        0.5
        * np.exp(VELOCITY_M_PER_YEAR * x_m / (2.0 * dispersion))
        * (
            np.exp(-root * x_m / (2.0 * dispersion)) * erfc((x_m - root * FINAL_TIME_YEARS) / scale)
            + np.exp(root * x_m / (2.0 * dispersion))
            * erfc((x_m + root * FINAL_TIME_YEARS) / scale)
        )
    )


def error_metrics(profile: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    difference = np.asarray(profile) - np.asarray(reference)
    return {
        "l2": float(np.linalg.norm(difference)),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "l1_mean": float(np.mean(np.abs(difference))),
        "linf": float(np.max(np.abs(difference))),
    }


def main() -> None:
    x = np.arange(1, NGRID_NODES) * GRID_SPACING_M
    centers = x + GRID_SPACING_M / 2
    dense = np.linspace(0.0, LENGTH_M, 601)
    reference = analytical_solution(x)
    center_reference = analytical_solution(centers)
    profiles, rows = ({}, [])
    for method in METHODS:
        for cfl in PAPER_CFL_VALUES:
            profile, metadata = run_method(method, cfl)
            profiles[_profile_key(method, cfl)] = profile
            rows.append(
                {
                    **metadata,
                    **{f"paper_node_{k}": v for k, v in error_metrics(profile, reference).items()},
                    **{
                        f"cell_center_{k}": v
                        for k, v in error_metrics(profile, center_reference).items()
                    },
                }
            )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "paper_figure6_data.npz",
        x_paper_nodes_m=x,
        x_modflow_cell_centers_m=centers,
        x_analytical_dense_m=dense,
        analytical_paper_nodes=reference,
        analytical_cell_centers=center_reference,
        analytical_dense=analytical_solution(dense),
        **profiles,
    )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "paper_figure6_metrics.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
