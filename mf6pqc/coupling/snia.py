"""Sequential non-iterative (SNIA) reactive-transport coupling."""

from __future__ import annotations

import time

import numpy as np

from mf6pqc.backends import initialize_modflow6
from mf6pqc.coupling.common import (
    build_standard_state,
    cache_basic_geometry,
    enforce_component_domains,
    finalize_results,
    get_calculated_density,
    get_coupling_time_step,
    log_progress,
    read_concentrations_from_modflow,
    run_reaction_step,
    save_time_step_results,
    should_run_reaction,
    simulation_has_time_remaining,
    solve_modflow_solutions,
    update_selected_output,
    validate_setup,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.state import StandardCouplingState
from mf6pqc.feedback import update_medium_properties, write_conductivity_for_step


def standard_time_step(sim, state: StandardCouplingState) -> None:
    """Advance one transport step and run chemistry when scheduled."""
    dt = get_coupling_time_step(state)
    sim.modflow_api.prepare_time_step(dt)
    write_conductivity_for_step(sim, state.current_k11, state.logical_step)
    density = get_calculated_density(sim) if sim.if_update_density else None
    solve_modflow_solutions(sim, state, density)
    sim.modflow_api.finalize_time_step()
    state.current_time = float(sim.modflow_api.get_current_time())

    if should_run_reaction(sim, state.logical_step):
        read_concentrations_from_modflow(
            state.concentration_variables, state.species_slices, state.transported
        )
        enforce_component_domains(
            state.transported,
            sim.components,
            state.species_slices,
            sim.signed_components,
        )
        reaction_start_time = state.last_reaction_time
        reaction_dt = state.current_time - reaction_start_time
        run_reaction_step(
            sim,
            state.transported,
            state.reacted,
            reaction_start_time,
            reaction_dt,
        )
        state.last_reaction_time = state.current_time
        update_selected_output(sim)
        write_concentrations_to_modflow(
            state.concentration_variables, state.species_slices, state.reacted
        )
        state.current_k11 = update_medium_properties(
            sim, state.current_k11, state.logical_step
        )
        save_time_step_results(sim, state.logical_step, state.current_time)
    state.logical_step += 1
    log_progress(
        state.current_time,
        state.end_time,
        state.logical_step,
        interval=sim.progress_interval,
    )


def run_standard(sim) -> None:
    """Run sequential non-iterative reactive transport to the TDIS end time."""
    validate_setup(sim)
    initialize_modflow6(sim)
    print("\n--- Starting reactive transport simulation (SNIA) ---")
    start = time.perf_counter()
    cache_basic_geometry(sim)
    state = build_standard_state(sim)
    while simulation_has_time_remaining(state.current_time, state.end_time):
        standard_time_step(sim, state)
    finalize_results(sim, state.logical_step, start)
