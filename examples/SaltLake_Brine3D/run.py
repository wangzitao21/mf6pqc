"""Run the idealized three-dimensional salt-lake brine-mining case."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = CASE_DIR.parents[1]
sys.path.insert(0, str(REPOSITORY_DIR))

from mf6pqc import MF6PQC  # noqa: E402
from mf6pqc.permeability import PowerLawUpdater  # noqa: E402

from case_config import (  # noqa: E402
    FACIES_MINERAL_VOLUME_FRACTIONS,
    INITIAL_POROSITY,
    K33_RATIO,
    MINERAL_MOLAR_VOLUMES_L_PER_MOL,
    PROFILES,
    cell_index,
    initial_hydraulic_conductivity,
    initial_porosity,
    kinetic_facies,
    potassium_grade_percent,
)
from modflow_model import build_model  # noqa: E402


SCENARIOS = ("feedback", "fixed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate halite-saturated canal recharge, evaporite dissolution, "
            "and brine pumping in a three-layer confined aquifer."
        )
    )
    parser.add_argument(
        "--profile",
        choices=tuple(PROFILES),
        default="highres",
        help=(
            "smoke: 252 cells/2 years; base: 1350 cells/30 years; "
            "highres: 5400 cells/30 years with a GSTools lognormal K field"
        ),
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        default="feedback",
        help="feedback updates porosity/K; fixed keeps the same reactions at fixed properties",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(12, os.cpu_count() or 1)),
        help="PhreeqcRM worker threads",
    )
    parser.add_argument(
        "--no-density",
        action="store_true",
        help="disable Pitzer-density feedback to MODFLOW BUY",
    )
    return parser.parse_args()


def _component_dictionary(
    components: list[str], values: np.ndarray, nxyz: int | None = None
) -> dict[str, float]:
    values = np.asarray(values, dtype=float).ravel()
    if nxyz is None:
        return {name: float(values[index]) for index, name in enumerate(components)}
    return {
        name: float(values[index * nxyz]) for index, name in enumerate(components)
    }


def main() -> None:
    args = parse_args()
    if args.threads <= 0:
        raise ValueError("--threads must be positive")

    profile = PROFILES[args.profile]
    feedback_enabled = args.scenario == "feedback"
    density_enabled = not args.no_density
    run_label = f"{profile.name}_{args.scenario}"
    if not density_enabled:
        run_label += "_constant_density"

    workspace = CASE_DIR / "simulation" / run_label
    output_dir = CASE_DIR / "output" / run_label
    porosity = initial_porosity(profile)
    hydraulic_conductivity = initial_hydraulic_conductivity(profile)
    facies = kinetic_facies(profile)

    simulator = MF6PQC(
        case_name=f"SaltLake_Brine3D_{run_label}",
        nxyz=profile.nxyz,
        nthreads=args.threads,
        temperature=25.0,
        pressure=2.0,
        porosity=porosity,
        saturation=1.0,
        density=1.20,
        d0=1.0e-9 * 86_400.0,
        print_chemistry_mask=0,
        componentH2O=False,
        solution_density_volume=False,
        db_path=str(CASE_DIR.parent / "Hamann2015" / "input_data" / "pitzer.dat"),
        pqi_path=str(CASE_DIR / "input_data" / "input.pqi"),
        modflow_dll_path=str(
            REPOSITORY_DIR / "bin" / "mf6.7.0" / "libmf6.dll"
        ),
        workspace=str(workspace),
        output_dir=str(output_dir),
        if_update_porosity_K=feedback_enabled,
        if_update_density=density_enabled,
        use_phreeqc_calculated_density=density_enabled,
        save_steps=profile.save_steps,
        progress_interval=profile.steps_per_year,
        fail_on_nonconvergence=True,
        permeability_updater=PowerLawUpdater(n=profile.permeability_exponent),
        k33_ratio=K33_RATIO,
        mineral_molar_volumes=MINERAL_MOLAR_VOLUMES_L_PER_MOL,
    )

    completed = False
    try:
        initial = simulator.setup(
            ic_map={
                "solution": 0,
                "equilibrium_phases": 1,
                "exchange": 1,
                "kinetics": facies,
            }
        )
        channel = simulator.get_initial_concentrations(1)
        components = simulator.get_components()

        # Charge is PHREEQC's numerical residual, not an independently
        # conserved transported solute. GWT boundary concentrations must be
        # nonnegative, so set its tiny roundoff residual exactly to zero.
        if "Charge" in components:
            charge_index = components.index("Charge")
            initial[
                charge_index * profile.nxyz : (charge_index + 1) * profile.nxyz
            ] = 0.0
            channel[charge_index] = 0.0

        build_model(
            workspace=workspace,
            mf6_executable=REPOSITORY_DIR / "bin" / "mf6.7.0" / "mf6.exe",
            profile=profile,
            species=components,
            initial_concentrations=initial,
            channel_concentrations=channel,
            porosity=porosity,
            hydraulic_conductivity=hydraulic_conductivity,
            density_feedback=density_enabled,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "initial_porosity.npy", porosity)
        np.save(output_dir / "initial_K.npy", hydraulic_conductivity)
        np.save(output_dir / "kinetic_facies.npy", facies)
        np.save(output_dir / "cell_volumes_m3.npy", profile.cell_volumes_m3.ravel())

        initial_selected = {
            str(name): float(simulator.selected_output[index, 0])
            for index, name in enumerate(simulator.headings)
        }
        metadata = {
            "schema_version": 1,
            "profile": profile.name,
            "scenario": args.scenario,
            "density_feedback": density_enabled,
            "units": {
                "length": "m",
                "time": "day",
                "flow": "m3/day",
                "hydraulic_conductivity": "m/day",
                "aqueous_concentration": "mol/L",
                "mineral_inventory": "mol/L representative bulk volume",
                "density": "kg/L",
            },
            "grid": {
                "nlay": profile.nlay,
                "nrow": profile.nrow,
                "ncol": profile.ncol,
                "delr_m": profile.delr,
                "delc_m": profile.delc,
                "layer_thicknesses_m": list(profile.layer_thicknesses),
                "top_m": profile.top,
                "botm_m": profile.botm.tolist(),
            },
            "time": {
                "years": profile.years,
                "days_per_year": 365.0,
                "steps_per_year": profile.steps_per_year,
                "total_steps": profile.total_steps,
                "save_steps": profile.save_steps,
            },
            "boundaries": {
                "channel_cells": [list(cell) for cell in profile.channel_cells],
                "channel_flat_indices": [
                    cell_index(profile, cell) for cell in profile.channel_cells
                ],
                "channel_stage_m": 35.0,
                "channel_width_m": profile.channel_width_m,
                "channel_conductance_per_cell_m2_per_day": (
                    profile.channel_conductance_per_cell_m2_per_day
                ),
                "well_cells": [list(cell) for cell in profile.well_cells],
                "well_flat_indices": [
                    cell_index(profile, cell) for cell in profile.well_cells
                ],
                "well_rate_each_m3_per_day": profile.well_rate_m3_per_day,
                "total_pumping_m3_per_day": (
                    profile.well_count * profile.well_rate_m3_per_day
                ),
            },
            "components": components,
            "initial_components_mol_per_litre": _component_dictionary(
                components, initial, profile.nxyz
            ),
            "channel_components_mol_per_litre": _component_dictionary(
                components, channel
            ),
            "initial_selected_output": initial_selected,
            "initial_porosity": INITIAL_POROSITY,
            "k33_to_k11_ratio": K33_RATIO,
            "initial_hydraulic_conductivity": {
                "distribution": "three-dimensional lognormal Gaussian random field",
                "generator": "GSTools Gaussian covariance + SRF",
                "seed": profile.random_field_seed,
                "log_standard_deviation": profile.log_k_standard_deviation,
                "correlation_lengths_xyz_m": list(
                    profile.k_correlation_lengths_m
                ),
                "layer_geometric_means_m_per_day": list(
                    profile.layer_geometric_mean_k_m_per_day
                ),
            },
            "permeability_exponent": profile.permeability_exponent,
            "permeability_law": (
                "K_new = K_initial * "
                f"(phi_new / phi_initial)^{profile.permeability_exponent:g}"
            ),
            "mineral_volume_fractions_by_facies": {
                str(key): value
                for key, value in FACIES_MINERAL_VOLUME_FRACTIONS.items()
            },
            "elemental_K_grade_percent_by_facies": {
                str(key): potassium_grade_percent(value)
                for key, value in FACIES_MINERAL_VOLUME_FRACTIONS.items()
            },
            "software_paths": {
                "database": str(
                    CASE_DIR.parent / "Hamann2015" / "input_data" / "pitzer.dat"
                ),
                "chemistry_input": str(CASE_DIR / "input_data" / "input.pqi"),
            },
        }
        (output_dir / "case_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        simulator.run(method="SNIA")
        simulator.save_results()
        completed = True
    finally:
        simulator.finalize()
        status = "completed" if completed else "failed"
        print(f"SaltLake_Brine3D {run_label} {status}.")


if __name__ == "__main__":
    main()
