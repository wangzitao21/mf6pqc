"""Run the ex016_Xie2015_B3 reactive-transport benchmark."""

from __future__ import annotations

import json
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


def normalize_solution(values, components, solution):
    values = np.asarray(values, dtype=float)
    ca = values[components.index("Ca")]
    if np.any(ca <= 0):
        raise ValueError("Concentration normalization requires positive Ca")
    result = values * (solution["ca+2"]["value"] / ca)
    for source, component in (
        ("ca+2", "Ca"),
        ("co3-2", "C"),
        ("so4-2", "S"),
        ("na+1", "Na"),
        ("al+3", "Al"),
        ("k+1", "K"),
    ):
        if source in solution and solution[source]["constraint"] != "charge":
            expected = solution[source]["value"]
            if not np.allclose(
                result[components.index(component)], expected, rtol=1e-8, atol=1e-12
            ):
                raise ValueError(f"Initial {component} differs from the specified total")
    if "fe+2" in solution:
        expected = solution["fe+2"]["value"] + solution["fe+3"]["value"]
        if not np.allclose(result[components.index("Fe")], expected, rtol=1e-8, atol=1e-12):
            raise ValueError("Initial total Fe differs from the specified input")
    return result


def main():
    parameters = json.loads((INPUT_DIR / "min3p_parameters.json").read_text())
    maximum = 0.1
    periods, saves, _ = schedule(
        parameters["final_years"], maximum, parameters["reference_times_years"]
    )
    n = 81
    phi = parameters["porosity"]
    conductivity = parameters["hydraulic_conductivity_m_per_s"] * 86400
    d0 = parameters["diffusion_m2_per_s"] * 86400
    widths = np.r_[0.0125, np.full(n - 2, 0.025), 0.0125]
    reactive = np.ones(n)
    entries = parameters["minerals"]
    reactions = tuple(
        KineticReaction(
            e["name"],
            e["stoichiometry"],
            reactive * e["rate_constant_mol_bulk_per_second"] * 86400,
            surface_exponent=e["surface_exponent"],
            minimum_amount=reactive * e["minimum_amount_mol_bulk"],
            saturation_index_heading="SI_" + e["name"],
            reference_amount=e["initial_amount_mol_bulk"] if e["surface_exponent"] else None,
        )
        for e in entries
    )
    feedback = FeedbackOptions(
        update_porosity_and_k=True,
        porosity_update_mask=reactive.astype(bool),
        mineral_molar_volumes={e["name"]: e["molar_volume_l_per_mol"] for e in entries},
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
            initial.reshape(sim.ncomps, n), sim.components, parameters["solutions"]["0"]
        )
        inlet = normalize_solution(
            sim.get_initial_concentrations(1), sim.components, parameters["solutions"]["1"]
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
