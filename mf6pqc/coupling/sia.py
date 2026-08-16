"""Source-based sequential iterative (SIA) coupling."""

from __future__ import annotations

import time

import numpy as np

from mf6pqc.backends import initialize_modflow6
from mf6pqc.constants import MIN_TIME_STEP
from mf6pqc.coupling.common import (
    _record_solver_failure,
    allocate_concentration_buffers,
    build_species_slices,
    build_time_step_schedule,
    cache_concentration_variables,
    cache_solution_iterations,
    cache_source_variables,
    enforce_component_domains,
    finalize_results,
    get_calculated_density,
    get_coupling_time_step,
    log_progress,
    read_concentrations_from_modflow,
    run_reaction_step,
    save_time_step_results,
    simulation_has_time_remaining,
    synchronize_phreeqcrm_solution,
    update_selected_output,
    validate_setup,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.state import SIACouplingState
from mf6pqc.exceptions import BackendError, ConvergenceError, CouplingError
from mf6pqc.feedback import (
    prepare_feedback,
    update_medium_properties,
    write_conductivity_for_step,
)
from mf6pqc.utils import get_gwt_model_name


def cache_cell_volume(sim, gwt_model_name: str) -> np.ndarray:
    """Return bulk cell volumes from one GWT discretization."""
    api = sim.modflow_api
    area = np.asarray(
        api.get_value_ptr(api.get_var_address("AREA", gwt_model_name, "DIS")),
        dtype=float,
    ).ravel()
    top = np.asarray(
        api.get_value_ptr(api.get_var_address("TOP", gwt_model_name, "DIS")),
        dtype=float,
    ).ravel()
    bottom = np.asarray(
        api.get_value_ptr(api.get_var_address("BOT", gwt_model_name, "DIS")),
        dtype=float,
    ).ravel()
    if not (area.size == top.size == bottom.size == sim.nxyz):
        raise BackendError("GWT cell geometry does not match nxyz")
    volume = area * (top - bottom)
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise BackendError("GWT contains non-positive or non-finite cell volumes")
    return volume


def build_sia_state(sim) -> SIACouplingState:
    """Create all runtime buffers and live pointers for SIA."""
    concentration_variables = cache_concentration_variables(
        sim.modflow_api, sim.components, sim.nxyz
    )
    current_k11 = prepare_feedback(sim)
    solution_iterations = cache_solution_iterations(sim.modflow_api)
    transported, reacted = allocate_concentration_buffers(sim.nxyz, sim.ncomps)
    reaction_input = np.empty_like(transported)
    previous_time = np.zeros_like(transported)
    previous_iteration = np.zeros_like(transported)
    source_rates = np.zeros_like(transported)
    candidate_source_rates = np.zeros_like(transported)
    species_slices = build_species_slices(sim.nxyz, sim.ncomps)
    source_variables = cache_source_variables(
        sim.modflow_api, sim.components, sim.nxyz
    )
    bulk_volume = cache_cell_volume(
        sim, get_gwt_model_name(sim.components[0])
    )
    previous_density = (
        get_calculated_density(sim) if sim.if_update_density else None
    )
    candidate_density = (
        previous_density.copy() if previous_density is not None else None
    )
    return SIACouplingState(
        concentration_variables=concentration_variables,
        species_slices=species_slices,
        transported=transported,
        reaction_input=reaction_input,
        reacted=reacted,
        previous_time_concentrations=previous_time,
        previous_iteration_concentrations=previous_iteration,
        source_rates=source_rates,
        candidate_source_rates=candidate_source_rates,
        concentration_difference=np.empty_like(transported),
        coupling_difference=np.empty_like(transported),
        source_difference=np.empty_like(transported),
        solution_iterations=solution_iterations,
        source_variables=source_variables,
        bulk_cell_volume=bulk_volume,
        mobile_water_volume=bulk_volume * sim.porosity * sim.saturation,
        current_time=float(sim.modflow_api.get_current_time()),
        end_time=float(sim.modflow_api.get_end_time()),
        logical_step=0,
        current_k11=current_k11,
        previous_density=previous_density,
        candidate_density=candidate_density,
        density_difference=(
            np.zeros_like(previous_density)
            if previous_density is not None
            else None
        ),
        picard_iteration=0,
        time_step_schedule=build_time_step_schedule(sim.modflow_api),
    )


def restore_concentrations(state: SIACouplingState) -> None:
    """Restore GWT concentrations to the previous time level."""
    for index, info in enumerate(state.concentration_variables.values()):
        pointer = info["ptr"]
        pointer[:] = state.previous_time_concentrations[
            state.species_slices[index]
        ].reshape(pointer.shape)


def apply_sia_sources(sim, state: SIACouplingState) -> None:
    """Write reaction and optional pure-water-sink source rates to GWT."""
    has_water_only_sinks = getattr(sim, "has_water_only_sinks", None)
    if has_water_only_sinks is None:
        has_water_only_sinks = bool(np.any(sim.water_only_sink_rates > 0.0))
    for index, component in enumerate(sim.components):
        species_slice = state.species_slices[index]
        pointer = state.source_variables[component]["ptr"]
        pointer[:] = state.source_rates[species_slice]
        if component != "H2O" and has_water_only_sinks:
            concentration = np.asarray(
                state.concentration_variables[component]["ptr"]
            ).ravel()
            pointer[:] += sim.water_only_sink_rates * concentration


def solve_modflow_picard(sim, state: SIACouplingState) -> None:
    """Solve every MODFLOW solution for one SIA Picard iterate."""
    for solution_id, iteration_pointer in state.solution_iterations.items():
        sim.modflow_api.prepare_solve(solution_id)
        apply_sia_sources(sim, state)
        if (
            sim.if_update_density
            and solution_id == 1
            and state.candidate_density is not None
        ):
            state.previous_density = (
                (1.0 - sim.sia_density_relaxation) * state.previous_density
                + sim.sia_density_relaxation * state.candidate_density
            )
            sim.density_ptr[:] = state.previous_density
        maximum = int(iteration_pointer[0])
        converged = False
        iterations = 0
        try:
            while iterations < maximum:
                converged = bool(sim.modflow_api.solve(solution_id))
                iterations += 1
                if converged:
                    break
        finally:
            sim.modflow_api.finalize_solve(solution_id)
        if not converged:
            _record_solver_failure(
                sim, solution_id, iterations, picard=True
            )


def build_reaction_input(state: SIACouplingState, dt: float) -> None:
    """Reconstruct the reaction base state for one source Picard iterate.

    The MODFLOW endpoint already contains the reaction source from the
    previous Picard iterate.  Feeding that endpoint directly to a full
    PhreeqcRM kinetic step applies the old reaction contribution a second
    time.  Equation (110) of Steefel and MacQuarrie (1996) instead requires
    the transport contribution with the old reaction term removed.  In GWT
    SRC units this is ``C_base = C_transport - q_reaction * dt / V_water``.
    """
    if dt <= MIN_TIME_STEP:
        raise CouplingError(f"SIA received a non-positive reaction step: {dt}")
    if (
        not np.all(np.isfinite(state.mobile_water_volume))
        or np.any(state.mobile_water_volume <= 0.0)
    ):
        raise CouplingError("SIA mobile-water volumes must be finite and positive")
    np.copyto(state.reaction_input, state.transported)
    for species_slice in state.species_slices:
        correction = state.source_difference[species_slice]
        np.copyto(correction, state.source_rates[species_slice])
        correction *= dt
        correction /= state.mobile_water_volume
        state.reaction_input[species_slice] -= correction


def update_sources(sim, state: SIACouplingState, dt: float) -> None:
    """Evaluate and relax the reaction-source fixed-point residual.

    ``source_difference`` deliberately stores the *unrelaxed* residual.  A
    small relaxation factor must not be able to manufacture convergence by
    making the applied source update artificially small.
    """
    relaxation = sim.sia_source_relaxation
    if dt <= MIN_TIME_STEP:
        state.source_rates.fill(0.0)
        state.candidate_source_rates.fill(0.0)
        state.source_difference.fill(0.0)
        np.subtract(
            state.reacted,
            state.transported,
            out=state.coupling_difference,
        )
        apply_sia_sources(sim, state)
        return
    for species_slice in state.species_slices:
        calculated = state.candidate_source_rates[species_slice]
        np.subtract(
            state.reacted[species_slice],
            state.reaction_input[species_slice],
            out=calculated,
        )
        calculated *= state.mobile_water_volume / dt
        np.subtract(
            calculated,
            state.source_rates[species_slice],
            out=state.source_difference[species_slice],
        )
    np.subtract(
        state.reacted,
        state.transported,
        out=state.coupling_difference,
    )
    state.source_rates += relaxation * state.source_difference
    apply_sia_sources(sim, state)


def update_sources_from_instantaneous_rates(
    sim, state: SIACouplingState, target_time: float
) -> None:
    """Evaluate a paper-style instantaneous reaction source at ``n + 1``.

    Steefel and MacQuarrie's equation (108) places ``R^(n+1,m)`` directly on
    the transport right-hand side.  A full PhreeqcRM reaction map instead
    returns the reaction integrated over an entire coupling interval.  The
    two are different for finite kinetic time steps.  This optional callback
    path is intended for stateless rate-law verification problems where the
    component rate is available explicitly.

    The callback contract is ``rate(components, C, time) -> dC/dt``.  ``C``
    is a read-only array shaped ``(n_components, n_cells)`` and the returned
    rates must have the same shape and use model-time units.
    """
    evaluator = sim.sia_rate_evaluator
    concentrations = state.transported.reshape(sim.ncomps, sim.nxyz).copy()
    concentrations.setflags(write=False)
    try:
        evaluated = evaluator(
            tuple(sim.components), concentrations, float(target_time)
        )
    except Exception as exc:
        raise CouplingError("sia_rate_evaluator failed") from exc
    rates = np.asarray(evaluated, dtype=float)
    expected_shape = (sim.ncomps, sim.nxyz)
    if rates.shape != expected_shape:
        raise CouplingError(
            "sia_rate_evaluator returned shape "
            f"{rates.shape}; expected {expected_shape}"
        )
    if not np.all(np.isfinite(rates)):
        raise CouplingError("sia_rate_evaluator returned non-finite rates")

    candidate = state.candidate_source_rates.reshape(expected_shape)
    candidate[:] = rates * state.mobile_water_volume[np.newaxis, :]
    np.subtract(
        state.candidate_source_rates,
        state.source_rates,
        out=state.source_difference,
    )
    np.copyto(state.reacted, state.transported)
    state.coupling_difference.fill(0.0)
    state.source_rates += (
        sim.sia_source_relaxation * state.source_difference
    )
    apply_sia_sources(sim, state)


def check_picard_convergence(sim, state: SIACouplingState) -> bool:
    """Check iterate change, transport-reaction closure, and source residual."""
    np.subtract(
        state.transported,
        state.previous_iteration_concentrations,
        out=state.concentration_difference,
    )
    np.abs(state.concentration_difference, out=state.concentration_difference)
    if state.picard_iteration == 0:
        return False
    iteration_ok = np.all(
        state.concentration_difference
        <= sim.sia_atol + sim.sia_rtol * np.abs(state.transported)
    )
    if not iteration_ok:
        return False

    coupling_scale = np.maximum(
        np.abs(state.transported), np.abs(state.reacted)
    )
    coupling_ok = np.all(
        np.abs(state.coupling_difference)
        <= sim.sia_atol + sim.sia_rtol * coupling_scale
    )
    if not coupling_ok:
        return False

    if state.density_difference is not None:
        density_scale = np.maximum(
            np.abs(state.candidate_density), np.abs(state.previous_density)
        )
        density_ok = np.all(
            np.abs(state.density_difference)
            <= sim.sia_atol + sim.sia_rtol * density_scale
        )
        if not density_ok:
            return False

    absolute_rate_tolerance = (
        sim.sia_atol * state.mobile_water_volume / state.current_dt
    )
    return all(
        np.all(
            np.abs(state.source_difference[species_slice])
            <= absolute_rate_tolerance
            + sim.sia_rtol
            * np.maximum(
                np.abs(state.candidate_source_rates[species_slice]),
                np.abs(
                    state.candidate_source_rates[species_slice]
                    - state.source_difference[species_slice]
                ),
            )
        )
        for species_slice in state.species_slices
    )


def picard_residual_summary(state: SIACouplingState) -> dict[str, float]:
    """Return compact absolute residual diagnostics for the current iterate."""
    summary = {
        "max_iteration_concentration": float(
            np.max(np.abs(state.concentration_difference))
        ),
        "max_transport_reaction_closure": float(
            np.max(np.abs(state.coupling_difference))
        ),
        "max_source_rate": float(np.max(np.abs(state.source_difference))),
    }
    if state.density_difference is not None:
        summary["max_density"] = float(
            np.max(np.abs(state.density_difference))
        )
    return summary


def run_picard_iteration(
    sim, state: SIACouplingState, reaction_start_time: float, dt: float
) -> bool:
    """Execute one source-correction Picard iteration."""
    restore_concentrations(state)
    solve_modflow_picard(sim, state)
    read_concentrations_from_modflow(
        state.concentration_variables, state.species_slices, state.transported
    )
    enforce_component_domains(
        state.transported,
        sim.components,
        state.species_slices,
        sim.signed_components,
    )
    if sim.sia_rate_evaluator is not None:
        update_sources_from_instantaneous_rates(
            sim, state, reaction_start_time + dt
        )
    else:
        build_reaction_input(state, dt)
        enforce_component_domains(
            state.reaction_input,
            sim.components,
            state.species_slices,
            sim.signed_components,
        )
        sim.phreeqc_rm.StateApply(1)
        run_reaction_step(
            sim, state.reaction_input, state.reacted, reaction_start_time, dt
        )
        if sim.if_update_density:
            if not sim.use_phreeqc_calculated_density:
                update_selected_output(sim)
            state.candidate_density = get_calculated_density(sim)
            np.subtract(
                state.candidate_density,
                state.previous_density,
                out=state.density_difference,
            )
        update_sources(sim, state, dt)
    if check_picard_convergence(sim, state):
        # The two endpoints agree to tolerance.  Commit the chemistry endpoint
        # so GWT and PhreeqcRM start the next logical step from one state.
        write_concentrations_to_modflow(
            state.concentration_variables,
            state.species_slices,
            state.reacted,
        )
        return True
    np.copyto(state.previous_iteration_concentrations, state.transported)
    return False


def run_picard_loop(
    sim, state: SIACouplingState, reaction_start_time: float, dt: float
) -> int:
    """Iterate transport/reaction source correction to configured tolerances."""
    for iteration in range(sim.sia_max_iterations):
        state.picard_iteration = iteration
        if run_picard_iteration(sim, state, reaction_start_time, dt):
            count = iteration + 1
            sim.sia_diagnostics.append(
                {
                    "time_days": float(reaction_start_time + dt),
                    "iterations": count,
                    "converged": True,
                    **picard_residual_summary(state),
                }
            )
            return count
    residuals = picard_residual_summary(state)
    closure_by_component = {
        component: float(
            np.max(np.abs(state.coupling_difference[species_slice]))
        )
        for component, species_slice in zip(sim.components, state.species_slices)
    }
    worst_component = max(closure_by_component, key=closure_by_component.get)
    failure = {
        "time_days": float(reaction_start_time + dt),
        "iterations": int(sim.sia_max_iterations),
        **residuals,
        "closure_by_component": closure_by_component,
    }
    sim.sia_convergence_failures.append(failure)
    sim.sia_diagnostics.append({**failure, "converged": False})
    message = (
        "SIA failed to converge: "
        f"time={reaction_start_time + dt:.6g} days, "
        f"iterations={sim.sia_max_iterations}, "
        f"max_dC_iter={residuals['max_iteration_concentration']:.3e}, "
        f"max_closure={residuals['max_transport_reaction_closure']:.3e}, "
        f"closure_component={worst_component}, "
        f"max_dsource={residuals['max_source_rate']:.3e}"
    )
    if sim.sia_fail_on_nonconvergence:
        raise ConvergenceError(message)
    print(f"Warning: {message}")
    # Continuing is a legacy, non-strict policy.  Keep the two backends on the
    # same last reaction endpoint even though the transport residual is above
    # tolerance; the recorded diagnostics make that degradation explicit.
    write_concentrations_to_modflow(
        state.concentration_variables,
        state.species_slices,
        state.reacted,
    )
    return sim.sia_max_iterations


def update_sia_after_step(
    sim, state: SIACouplingState, picard_iterations: int
) -> None:
    """Commit diagnostics and medium feedback after a converged SIA step."""
    update_selected_output(sim)
    state.current_k11 = update_medium_properties(
        sim, state.current_k11, state.logical_step
    )
    state.mobile_water_volume = (
        state.bulk_cell_volume * sim.porosity * sim.saturation
    )
    save_time_step_results(sim, state.logical_step, state.current_time)
    sim.sia_iterations.append(picard_iterations)
    state.logical_step += 1
    log_progress(
        state.current_time,
        state.end_time,
        state.logical_step,
        f"SIA iters={picard_iterations}",
        interval=sim.progress_interval,
    )


def sia_time_step(sim, state: SIACouplingState) -> None:
    """Advance one source-iterated SIA coupling step."""
    dt = get_coupling_time_step(state)
    if dt <= MIN_TIME_STEP:
        raise CouplingError(f"SIA received a non-positive coupling step: {dt}")
    sim.modflow_api.prepare_time_step(dt)
    write_conductivity_for_step(sim, state.current_k11, state.logical_step)
    sim.phreeqc_rm.StateSave(1)
    state_saved = True
    try:
        read_concentrations_from_modflow(
            state.concentration_variables,
            state.species_slices,
            state.previous_time_concentrations,
        )
        np.copyto(
            state.previous_iteration_concentrations,
            state.previous_time_concentrations,
        )
        state.source_rates.fill(0.0)
        apply_sia_sources(sim, state)
        state.current_dt = dt
        iterations = run_picard_loop(sim, state, state.current_time, dt)
        read_concentrations_from_modflow(
            state.concentration_variables,
            state.species_slices,
            state.transported,
        )
        enforce_component_domains(
            state.transported,
            sim.components,
            state.species_slices,
            sim.signed_components,
        )
        if sim.sia_rate_evaluator is not None:
            synchronize_phreeqcrm_solution(
                sim,
                state.transported,
                state.current_time + dt,
                preserve_transport_endpoint=True,
            )
        sim.phreeqc_rm.StateDelete(1)
        state_saved = False
        sim.modflow_api.finalize_time_step()
        state.current_time = float(sim.modflow_api.get_current_time())
        update_sia_after_step(sim, state, iterations)
    finally:
        if state_saved:
            sim.phreeqc_rm.StateDelete(1)


def run_sia(sim) -> None:
    """Run source-based sequential iterative coupling to the TDIS end time."""
    validate_setup(sim)
    initialize_modflow6(sim)
    print("\n--- Starting reactive transport simulation (SIA) ---")
    start = time.perf_counter()
    state = build_sia_state(sim)
    while simulation_has_time_remaining(state.current_time, state.end_time):
        sia_time_step(sim, state)
    finalize_results(sim, state.logical_step, start)
