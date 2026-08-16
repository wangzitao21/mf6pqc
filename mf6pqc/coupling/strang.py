"""Symmetric Strang transport-reaction splitting."""

from __future__ import annotations

import time

import numpy as np

from mf6pqc.backends import initialize_modflow6
from mf6pqc.constants import MIN_TIME_STEP
from mf6pqc.coupling.common import (
    build_standard_state,
    cache_basic_geometry,
    enforce_component_domains,
    finalize_results,
    get_calculated_density,
    log_progress,
    read_concentrations_from_modflow,
    run_reaction_step,
    save_time_step_results,
    simulation_has_time_remaining,
    solve_modflow_solutions,
    synchronize_phreeqcrm_solution,
    update_selected_output,
    validate_setup,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.state import StandardCouplingState
from mf6pqc.exceptions import CouplingError
from mf6pqc.feedback import update_medium_properties, write_conductivity_for_step


def validate_strang_schedule(schedule: np.ndarray) -> np.ndarray:
    """Validate and return logical durations for paired TDIS half-steps."""
    values = np.asarray(schedule, dtype=float).ravel()
    if values.size == 0 or values.size % 2:
        raise CouplingError(
            "Strang splitting requires a non-empty even number of MODFLOW "
            "TDIS steps"
        )
    pairs = values.reshape(-1, 2)
    equal = np.isclose(
        pairs[:, 0], pairs[:, 1], rtol=1.0e-10, atol=MIN_TIME_STEP
    )
    if not np.all(equal):
        pair_index = int(np.flatnonzero(~equal)[0])
        first_half, second_half = pairs[pair_index]
        raise CouplingError(
            "Strang splitting requires equal adjacent TDIS half-steps; "
            f"logical step {pair_index + 1} has {first_half!r} and "
            f"{second_half!r}"
        )
    return np.sum(pairs, axis=1)


def solve_transport_substep(
    sim,
    state: StandardCouplingState,
    dt: float,
    density: np.ndarray | None,
    current_k11: np.ndarray | None,
    logical_step: int,
    *,
    force_conductivity_write: bool = False,
) -> None:
    """Advance MODFLOW 6 by one transport half-step."""
    sim.modflow_api.prepare_time_step(dt)
    write_conductivity_for_step(
        sim,
        current_k11,
        logical_step,
        force=force_conductivity_write,
    )
    solve_modflow_solutions(sim, state, density)
    sim.modflow_api.finalize_time_step()
    state.current_time = float(sim.modflow_api.get_current_time())


def strang_time_step(sim, state: StandardCouplingState) -> None:
    """Advance ``T(dt/2) -> R(dt) -> T(dt/2)`` for one logical step."""
    index = state.transport_step
    schedule = state.time_step_schedule
    if index + 1 >= schedule.size:
        raise CouplingError(
            "Strang splitting requires pairs of equal-length MODFLOW TDIS steps"
        )
    first_half = float(schedule[index])
    second_half = float(schedule[index + 1])
    if not np.isclose(first_half, second_half, rtol=1.0e-10, atol=MIN_TIME_STEP):
        raise CouplingError(
            "Strang splitting requires equal adjacent TDIS half-steps; "
            f"got {first_half!r} and {second_half!r}"
        )
    reaction_dt = first_half + second_half
    logical_start_time = state.current_time

    density = get_calculated_density(sim) if sim.if_update_density else None
    solve_transport_substep(
        sim,
        state,
        first_half,
        density,
        state.current_k11,
        state.logical_step,
    )
    read_concentrations_from_modflow(
        state.concentration_variables, state.species_slices, state.transported
    )
    enforce_component_domains(
        state.transported,
        sim.components,
        state.species_slices,
        sim.signed_components,
    )

    run_reaction_step(
        sim,
        state.transported,
        state.reacted,
        logical_start_time,
        reaction_dt,
    )
    update_selected_output(sim)
    write_concentrations_to_modflow(
        state.concentration_variables, state.species_slices, state.reacted
    )

    # Reaction-owned medium properties belong to the midpoint state and must
    # affect the second transport half-step for a true T/2 -> R -> T/2
    # composition.  Deferring them to the logical endpoint is a lagged SNIA
    # feedback, not Strang splitting.
    state.current_k11 = update_medium_properties(
        sim, state.current_k11, state.logical_step
    )
    density = get_calculated_density(sim) if sim.if_update_density else None
    solve_transport_substep(
        sim,
        state,
        second_half,
        density,
        state.current_k11,
        state.logical_step,
        force_conductivity_write=True,
    )
    read_concentrations_from_modflow(
        state.concentration_variables, state.species_slices, state.transported
    )
    enforce_component_domains(
        state.transported,
        sim.components,
        state.species_slices,
        sim.signed_components,
    )
    synchronize_phreeqcrm_solution(
        sim,
        state.transported,
        state.current_time,
        preserve_transport_endpoint=True,
    )

    save_time_step_results(sim, state.logical_step, state.current_time)
    state.logical_step += 1
    state.transport_step += 2
    log_progress(
        state.current_time,
        state.end_time,
        state.logical_step,
        "Strang",
        interval=sim.progress_interval,
    )


def run_strang(sim) -> None:
    """Run symmetric Strang splitting to the TDIS end time."""
    validate_setup(sim)
    initialize_modflow6(sim)
    print("\n--- Starting reactive transport simulation (Strang splitting) ---")
    start = time.perf_counter()
    cache_basic_geometry(sim)
    state = build_standard_state(sim)
    logical_schedule = validate_strang_schedule(state.time_step_schedule)
    if sim.save_steps is not None and max(sim.save_steps) > logical_schedule.size:
        raise CouplingError(
            "save_steps contains a logical Strang step beyond the paired TDIS "
            f"schedule: {max(sim.save_steps)} > {logical_schedule.size}"
        )
    while simulation_has_time_remaining(state.current_time, state.end_time):
        strang_time_step(sim, state)
    finalize_results(sim, state.logical_step, start)
