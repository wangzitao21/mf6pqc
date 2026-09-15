"""Sequential non-iterative (SNIA) reactive-transport coupling."""

from __future__ import annotations

import logging
import time

from mf6pqc.backends import initialize_modflow6
from mf6pqc.coupling.common import (
    StandardCouplingState,
    advance_to_end,
    build_standard_state,
    cache_basic_geometry,
    commit_reaction_concentrations,
    enforce_component_domains,
    finalize_results,
    get_calculated_density,
    get_coupling_time_step,
    read_concentrations_from_modflow,
    run_reaction_step,
    save_time_step_results,
    should_run_reaction,
    solve_modflow_solutions,
    update_selected_output,
    validate_setup,
)
from mf6pqc.feedback import update_medium_properties, write_conductivity_for_step

_logger = logging.getLogger(__name__)


def standard_time_step(sim, state: StandardCouplingState) -> None:
    """Advance one transport step and run chemistry when scheduled."""
    dt = get_coupling_time_step(state)
    sim.modflow_api.prepare_time_step(dt)
    write_conductivity_for_step(sim, state.current_k11, state.logical_step)
    density = get_calculated_density(sim) if sim.if_update_density else None
    solve_modflow_solutions(sim, state, density)
    sim.modflow_api.finalize_time_step()
    state.current_time = float(sim.modflow_api.get_current_time())

    hooks = getattr(sim, "_coupling_hooks", None)
    on_transport = None if hooks is None else hooks.on_transport
    react = should_run_reaction(sim, state.logical_step)
    if react or on_transport is not None:
        read_concentrations_from_modflow(
            state.concentration_variables, state.species_slices, state.transported
        )
    if on_transport is not None:
        on_transport(sim, state, dt)
    if react:
        enforce_component_domains(
            state.transported,
            sim.components,
            state.species_slices,
            sim.signed_components,
            nonnegative_slices=getattr(state, "nonnegative_slices", None),
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
        if hooks is not None and hooks.on_reaction is not None:
            hooks.on_reaction(sim, state)
        state.current_k11 = update_medium_properties(sim, state.current_k11, state.logical_step)
        commit_reaction_concentrations(sim, state)
        save_time_step_results(
            sim, state.logical_step, state.current_time, current_k11=state.current_k11
        )
    state.logical_step += 1


def run_standard(sim) -> None:
    """Run sequential non-iterative reactive transport to the TDIS end time."""
    validate_setup(sim)
    initialize_modflow6(sim)
    _logger.info("\n--- Starting reactive transport simulation (SNIA) ---")
    start = time.perf_counter()
    cache_basic_geometry(sim)
    state = build_standard_state(sim)
    advance_to_end(sim, state, standard_time_step)
    finalize_results(sim, state.logical_step, start)
