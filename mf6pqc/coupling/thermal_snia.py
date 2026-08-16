"""Explicit GWF-GWT-GWE-VSC sequential non-iterative coupling."""

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
    simulation_has_time_remaining,
    solve_modflow_solutions,
    update_selected_output,
    validate_setup,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.state import StandardCouplingState
from mf6pqc.energy import (
    capture_flow_inputs,
    capture_flow_response,
    save_energy_time_step_results,
    setup_energy_coupling,
)
from mf6pqc.exceptions import ConfigurationError
from mf6pqc.feedback import update_medium_properties, write_conductivity_for_step


def thermal_time_step(sim, state: StandardCouplingState) -> None:
    """Advance one explicit flow/transport/energy/reaction step.

    The current GWE temperature and reaction-updated reference K feed the flow
    solve.  GWE then advances temperature, and that new temperature is sent to
    PhreeqcRM for the reaction calculation.  As with ordinary SNIA, feedback
    produced at the end of a step is used by flow in the following step.
    """
    dt = get_coupling_time_step(state)
    reaction_start_time = state.current_time
    sim.modflow_api.prepare_time_step(dt)
    capture_flow_inputs(sim, state.current_k11)
    write_conductivity_for_step(sim, state.current_k11, state.logical_step)
    density = get_calculated_density(sim) if sim.if_update_density else None
    solve_modflow_solutions(sim, state, density)
    capture_flow_response(sim)
    sim.modflow_api.finalize_time_step()
    state.current_time = float(sim.modflow_api.get_current_time())

    read_concentrations_from_modflow(
        state.concentration_variables, state.species_slices, state.transported
    )
    enforce_component_domains(
        state.transported,
        sim.components,
        state.species_slices,
        sim.signed_components,
    )
    # run_reaction_step synchronizes the post-GWE temperature before RunCells.
    run_reaction_step(
        sim, state.transported, state.reacted, reaction_start_time, dt
    )
    update_selected_output(sim)
    write_concentrations_to_modflow(
        state.concentration_variables, state.species_slices, state.reacted
    )
    state.current_k11 = update_medium_properties(
        sim, state.current_k11, state.logical_step
    )
    save_time_step_results(sim, state.logical_step, state.current_time)
    save_energy_time_step_results(sim, state.logical_step)
    state.logical_step += 1
    log_progress(
        state.current_time,
        state.end_time,
        state.logical_step,
        suffix="GWE/VSC",
        interval=sim.progress_interval,
    )


def run_thermal_snia(sim) -> None:
    """Run the opt-in explicit GWE/VSC reactive-transport algorithm."""
    validate_setup(sim)
    if not sim.energy_enabled:
        raise ConfigurationError(
            "ThermalSNIA requires energy_enabled=True and a MODFLOW GWE model"
        )
    initialize_modflow6(sim)
    print("\n--- Starting reactive transport simulation (ThermalSNIA) ---")
    start = time.perf_counter()
    cache_basic_geometry(sim)
    setup_energy_coupling(sim)
    state = build_standard_state(sim)
    while simulation_has_time_remaining(state.current_time, state.end_time):
        thermal_time_step(sim, state)
    finalize_results(sim, state.logical_step, start)


__all__ = ["run_thermal_snia", "thermal_time_step"]
