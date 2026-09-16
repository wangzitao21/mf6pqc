"""Run the ex016_Xie2015_B3 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex016_Xie2015_B3.modflow_model import build_model
from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    FeedbackOptions,
    ImplicitOptions,
    KineticReaction,
    OutputOptions,
    SimulationConfig,
)

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

POROSITY = 0.35
HYDRAULIC_CONDUCTIVITY = 0.000116 * 86400
DIFFC = 0.0
FINAL_YEARS = 300
MAXIMUM_STEP_YEARS = 0.1
PROFILE_YEARS = (10, 100, 300)
MINERAL_MOLAR_VOLUMES = {
    "Calcite": 0.03693335793357933,
    "Gypsum": 0.0742121551724138,
    "Ferrihydrite": 0.023990391874270812,
    "Jarosite": 0.154628125,
    "Gibbsite": 0.033193063829787234,
    "Siderite": 0.029256666666666663,
}
MINERALS = {
    "Calcite": {
        "stoichiometry": {"Ca": 1, "C": 1, "O": 3},
        "rate_constant": 5e-08,
        "initial_volume_fraction": 0.22,
        "surface_exponent": 2 / 3,
    },
    "Gypsum": {
        "stoichiometry": {"Ca": 1, "S": 1, "O": 6, "H": 4},
        "rate_constant": 5e-08,
        "initial_volume_fraction": 0.0,
        "surface_exponent": 0,
    },
    "Ferrihydrite": {
        "stoichiometry": {"Fe": 1, "O": 3, "H": 3},
        "rate_constant": 5e-09,
        "initial_volume_fraction": 0.0,
        "surface_exponent": 0,
    },
    "Jarosite": {
        "stoichiometry": {"K": 1, "Fe": 3, "S": 2, "O": 14, "H": 6},
        "rate_constant": 5e-09,
        "initial_volume_fraction": 0.0,
        "surface_exponent": 0,
    },
    "Gibbsite": {
        "stoichiometry": {"Al": 1, "O": 3, "H": 3},
        "rate_constant": 5e-10,
        "initial_volume_fraction": 0.05,
        "surface_exponent": 2 / 3,
    },
    "Siderite": {
        "stoichiometry": {"Fe": 1, "C": 1, "O": 3},
        "rate_constant": 5e-09,
        "initial_volume_fraction": 0.05,
        "surface_exponent": 2 / 3,
    },
}
SOLUTION_TOTALS = (
    {
        "Ca": 0.0004708717,
        "C": 0.002192803,
        "S": 0.000169512,
        "Al": 2.789895e-07,
        "K": 1e-05,
        "Fe": 6.61792942e-06,
    },
    {
        "Ca": 0.0001,
        "C": 0.01,
        "S": 0.1,
        "Na": 0.09092,
        "Al": 0.0143,
        "K": 7.67e-05,
        "Fe": 0.0223000114,
    },
)


def schedule(years, maximum, targets):
    if not np.isfinite(years) or not np.isfinite(maximum) or min(years, maximum) <= 0:
        raise ValueError("Duration and maximum time step must be positive and finite")
    ends = sorted({v for v in [1.0, *targets, years] if 0 < v <= years})
    periods, steps, save_steps = ([], [], [])
    previous = 0.0

    def append_period(duration, count, multiplier):
        weights = multiplier ** (np.arange(count, dtype=float) - count + 1)
        durations = duration * 365 * weights / weights.sum()
        durations[-1] += duration * 365 - durations.sum()
        periods.append((duration * 365, count, multiplier))
        steps.extend(durations)

    for end in ends:
        duration = end - previous
        if previous == 0:
            count = max(int(np.ceil(duration / maximum)), 100)
            multiplier = 1.12
            while (1 - 1 / multiplier) / (1 - multiplier ** (-count)) * duration > maximum:
                multiplier = np.sqrt(multiplier)
                count *= 2
            append_period(duration, count, multiplier)
        else:
            next_step = steps[-1] / 365 * 1.12
            ramp, count = (0.0, 0)
            while next_step < maximum and ramp + next_step < duration:
                ramp += next_step
                count += 1
                next_step *= 1.12
            if count:
                append_period(ramp, count, 1.12)
            remaining = duration - ramp
            append_period(remaining, max(1, int(np.ceil(remaining / maximum))), 1.0)
        save_steps.append(len(steps))
        previous = end
    times = np.cumsum(steps)
    stride = max(1, int(np.ceil(len(steps) / 300)))
    save_steps = sorted({*save_steps, *range(stride, len(steps) + 1, stride)})
    return (periods, save_steps, times)


def normalize_solution(values, components, totals):
    values = np.asarray(values, dtype=float)
    ca = values[components.index("Ca")]
    if np.any(ca <= 0):
        raise ValueError("Concentration normalization requires positive Ca")
    result = values * (totals["Ca"] / ca)
    for component, expected in totals.items():
        if not np.allclose(result[components.index(component)], expected, rtol=1e-8, atol=1e-12):
            raise ValueError(f"Initial {component} differs from the specified total")
    return result


def main():
    periods, saves, _ = schedule(FINAL_YEARS, MAXIMUM_STEP_YEARS, PROFILE_YEARS)
    n = 81
    phi = POROSITY
    conductivity = HYDRAULIC_CONDUCTIVITY
    d0 = DIFFC
    widths = np.r_[0.0125, np.full(n - 2, 0.025), 0.0125]
    reactive = np.ones(n)
    reactions = tuple(
        KineticReaction(
            name,
            mineral["stoichiometry"],
            reactive * mineral["rate_constant"] * 86400,
            surface_exponent=mineral["surface_exponent"],
            minimum_amount=reactive * 1e-10 / MINERAL_MOLAR_VOLUMES[name],
            saturation_index_heading="SI_" + name,
            reference_amount=(
                mineral["initial_volume_fraction"] / MINERAL_MOLAR_VOLUMES[name]
                if mineral["surface_exponent"]
                else None
            ),
        )
        for name, mineral in MINERALS.items()
    )
    feedback = FeedbackOptions(
        update_porosity_and_k=True,
        porosity_update_mask=reactive.astype(bool),
        mineral_molar_volumes=MINERAL_MOLAR_VOLUMES,
        vertical_to_horizontal_k_ratio=1.0,
        fail_on_porosity_clipping=True,
        boundary_conductance_updates={
            "BUSHUI": {"cell_index": 0, "distance": widths[0] / 2},
            "GHB_RIGHT": {"cell_index": -1, "distance": widths[-1] / 2},
        },
    )
    config = SimulationConfig(
        case_name=CASE_DIR.name,
        nxyz=n,
        nthreads=4,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(pressure_atm=1.0, porosity=phi, free_water_diffusion_model_units=d0),
        feedback=feedback,
        output=OutputOptions(save_steps=saves),
        implicit=ImplicitOptions(
            reactions=reactions,
            dense_limit=1024,
            derivative_refresh=1,
            chemical_jacobian="species",
            maximum_iterations=80,
            porosity_coupling="lagged",
            predict_porosity=False,
        ),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(config) as sim:
        initial = sim.setup(ic_map={"solution": 0, "kinetics": 1})
        initial = normalize_solution(
            initial.reshape(sim.ncomps, n), sim.components, SOLUTION_TOTALS[0]
        )
        inlet = normalize_solution(
            sim.get_initial_concentrations(1), sim.components, SOLUTION_TOTALS[1]
        )
        sim.phreeqc_rm.SetConcentrations(initial.ravel())
        sim.phreeqc_rm.SetTimeStep(0)
        sim.phreeqc_rm.RunCells()
        from mf6pqc.coupling.common import update_selected_output

        update_selected_output(sim)
        sim.results[0] = sim.selected_output
        initial = initial.ravel()
        build_model(
            workspace=WORKSPACE,
            species=sim.get_components(),
            initial_concentrations=initial,
            inflow_concentrations=inlet,
            nlay=1,
            nrow=1,
            ncol=n,
            delr=widths,
            delc=[1.0],
            top=1.0,
            botm=0.0,
            period_data=periods,
            porosity=phi,
            hydraulic_conductivity=np.full((1, 1, n), conductivity),
            vertical_conductivity_ratio=1.0,
            initial_head=0.0,
            inlet_head=0.007,
            outlet_head=0.0,
            alh=0.0,
            ath1=0.0,
            diffc=np.cbrt(phi) * d0,
            boundary_distance=widths[0] / 2,
        )
        sim.run("Implicit")
        sim.save_results()
        print(f"B3 complete: {OUTPUT_DIR}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
