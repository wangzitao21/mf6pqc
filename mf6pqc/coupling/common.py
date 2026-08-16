"""Shared primitives for all MF6PQC coupling algorithms."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from mf6pqc.constants import (
    DENSITY_SCALE,
    MIN_CONCENTRATION,
    MIN_TIME_STEP,
    SECONDS_PER_DAY,
)
from mf6pqc.coupling.state import StandardCouplingState
from mf6pqc.exceptions import BackendError, ConvergenceError, CouplingError
from mf6pqc.feedback import prepare_feedback
from mf6pqc.utils import get_gwt_model_name, get_species_slice


def validate_setup(sim) -> None:
    """Require successful chemistry setup before starting a coupled run."""
    if not sim.is_setup:
        raise CouplingError("setup() must complete before a coupling run")
    if sim.phreeqc_rm is None:
        raise CouplingError("The PhreeqcRM backend is not available")


def cache_basic_geometry(sim) -> None:
    """Cache GWF geometry used by saturation-aware future extensions."""
    sim.head_addr = sim.modflow_api.get_var_address("X", sim.flow_model_name)
    model = sim.sim.get_model(sim.flow_model_name)
    sim.botm_arr = np.asarray(model.dis.bot.values).ravel()
    sim.top_arr = np.asarray(model.dis.top.values).ravel()
    if sim.botm_arr.size != sim.nxyz or sim.top_arr.size != sim.nxyz:
        raise BackendError(
            "GWF geometry does not match nxyz: "
            f"top={sim.top_arr.size}, bottom={sim.botm_arr.size}, nxyz={sim.nxyz}"
        )
    sim.cell_thick = sim.top_arr - sim.botm_arr
    if np.any(sim.cell_thick <= 0.0):
        raise BackendError("GWF contains cells with non-positive thickness")


def cache_concentration_variables(
    modflow_api, components: list[str], nxyz: int | None = None
) -> dict[str, dict[str, Any]]:
    """Cache live GWT concentration pointers in chemical-component order."""
    variables: dict[str, dict[str, Any]] = {}
    model_names: dict[str, str] = {}
    for component in components:
        model_name = get_gwt_model_name(component)
        normalized_name = model_name.casefold()
        if normalized_name in model_names:
            raise BackendError(
                "Chemical components map to the same MODFLOW model name: "
                f"{model_names[normalized_name]!r} and {component!r} -> {model_name!r}"
            )
        model_names[normalized_name] = component
        address = modflow_api.get_var_address("X", model_name)
        pointer = modflow_api.get_value_ptr(address)
        if nxyz is not None and pointer.size != nxyz:
            raise BackendError(
                f"GWT component {component!r} has {pointer.size} cells; expected {nxyz}"
            )
        variables[component] = {
            "address": address,
            "ptr": pointer,
            "shape": pointer.shape,
        }
        print(f"  - solute '{component}', shape={pointer.shape}")
    return variables


def cache_solution_iterations(modflow_api) -> dict[int, np.ndarray]:
    """Cache live IMS MXITER arrays for every MODFLOW solution."""
    count = int(modflow_api.get_subcomponent_count())
    if count <= 0:
        raise BackendError("MODFLOW 6 reported no numerical solutions")
    iterations: dict[int, np.ndarray] = {}
    for solution_id in range(1, count + 1):
        address = modflow_api.get_var_address("MXITER", f"SLN_{solution_id}")
        pointer = modflow_api.get_value_ptr(address)
        if pointer.size < 1 or int(pointer[0]) <= 0:
            raise BackendError(
                f"Invalid MXITER for MODFLOW solution {solution_id}: {pointer!r}"
            )
        iterations[solution_id] = pointer
    return iterations


def allocate_concentration_buffers(
    nxyz: int, ncomps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate transport and reaction buffers in component-major layout."""
    if nxyz <= 0 or ncomps <= 0:
        raise ValueError("nxyz and ncomps must be positive")
    transported = np.empty(nxyz * ncomps, dtype=float)
    reacted = np.empty_like(transported)
    return transported, reacted


def build_species_slices(nxyz: int, ncomps: int) -> tuple[slice, ...]:
    """Return one packed-buffer slice per chemical component."""
    return tuple(get_species_slice(nxyz, index) for index in range(ncomps))


def build_time_step_schedule(modflow_api) -> np.ndarray:
    """Expand immutable MODFLOW TDIS period data into explicit step lengths."""
    perlen = np.asarray(
        modflow_api.get_value("__INPUT__/SIM/TDIS/PERLEN"), dtype=float
    ).ravel()
    nstp = np.asarray(
        modflow_api.get_value("__INPUT__/SIM/TDIS/NSTP"), dtype=np.int64
    ).ravel()
    tsmult = np.asarray(
        modflow_api.get_value("__INPUT__/SIM/TDIS/TSMULT"), dtype=float
    ).ravel()
    if not (perlen.size == nstp.size == tsmult.size):
        raise BackendError("TDIS PERLEN, NSTP, and TSMULT lengths are inconsistent")
    if perlen.size == 0:
        raise BackendError("TDIS contains no stress periods")

    periods: list[np.ndarray] = []
    for period_length, step_count_raw, multiplier in zip(perlen, nstp, tsmult):
        step_count = int(step_count_raw)
        if (
            not np.isfinite(period_length)
            or not np.isfinite(multiplier)
            or period_length <= 0.0
            or step_count <= 0
            or multiplier <= 0.0
        ):
            raise ValueError(
                "TDIS period lengths, step counts, and multipliers must be finite and positive"
            )
        if np.isclose(multiplier, 1.0):
            steps = np.full(step_count, period_length / step_count, dtype=float)
        else:
            # Normalizing geometric weights avoids overflow in multiplier**nstp
            # for long schedules while preserving the exact period sum.
            exponents = np.arange(step_count, dtype=float)
            logarithms = exponents * np.log(multiplier)
            logarithms -= np.max(logarithms)
            weights = np.exp(logarithms)
            steps = period_length * weights / np.sum(weights)
        # Make the floating-point sum equal the declared period length.
        steps[-1] += period_length - float(np.sum(steps))
        if np.any(steps <= 0.0) or not np.all(np.isfinite(steps)):
            raise CouplingError("TDIS expansion produced a non-positive time step")
        periods.append(steps)
    return np.concatenate(periods)


def time_tolerance(end_time: float) -> float:
    """Floating-point tolerance used when comparing simulation times."""
    return np.finfo(float).eps * max(1.0, abs(end_time)) * 32.0


def simulation_has_time_remaining(current_time: float, end_time: float) -> bool:
    """Return whether a logical step remains, accounting for roundoff."""
    return current_time < end_time - time_tolerance(end_time)


def get_coupling_time_step(state) -> float:
    """Return the scheduled duration of the current logical coupling step."""
    index = state.logical_step
    schedule = state.time_step_schedule
    if index >= schedule.size:
        remaining = state.end_time - state.current_time
        if remaining <= time_tolerance(state.end_time):
            return 0.0
        raise CouplingError(
            "MODFLOW time-step schedule ended before the simulation end time: "
            f"remaining={remaining:.17g} days"
        )
    dt = float(schedule[index])
    remaining = state.end_time - state.current_time
    if dt > remaining and dt - remaining <= time_tolerance(state.end_time):
        return remaining
    return dt


def read_concentrations_from_modflow(
    concentration_variables: dict[str, dict[str, Any]],
    species_slices: tuple[slice, ...] | list[slice],
    destination: np.ndarray,
) -> None:
    """Pack live GWT arrays into a PhreeqcRM component-major buffer."""
    for index, info in enumerate(concentration_variables.values()):
        destination[species_slices[index]] = np.asarray(info["ptr"]).ravel()


def write_concentrations_to_modflow(
    concentration_variables: dict[str, dict[str, Any]],
    species_slices: tuple[slice, ...] | list[slice],
    source: np.ndarray,
) -> None:
    """Unpack a component-major reaction buffer into live GWT arrays."""
    for index, info in enumerate(concentration_variables.values()):
        pointer = info["ptr"]
        pointer[:] = source[species_slices[index]].reshape(pointer.shape)


def enforce_component_domains(
    concentrations: np.ndarray,
    components: list[str] | tuple[str, ...],
    species_slices: tuple[slice, ...] | list[slice],
    signed_components: frozenset[str] | set[str] | tuple[str, ...] = ("charge",),
) -> None:
    """Apply PHREEQC component domains after a transport solve.

    Element concentrations cannot be negative.  The ``Charge`` component is
    different: it stores charge imbalance in equivalents and its valid domain
    explicitly includes negative values.  Clipping it with the element totals
    changes alkalinity and can strongly perturb pH in otherwise dilute models.
    """
    if len(components) != len(species_slices):
        raise ValueError("components and species_slices must have equal length")
    signed = {component.casefold() for component in signed_components}
    for component, component_slice in zip(components, species_slices):
        if component.casefold() in signed:
            continue
        np.maximum(
            concentrations[component_slice],
            MIN_CONCENTRATION,
            out=concentrations[component_slice],
        )


def run_reaction_step(
    sim,
    transported: np.ndarray,
    reacted: np.ndarray,
    start_time: float,
    dt: float,
) -> None:
    """Advance PhreeqcRM reactions over ``[start_time, start_time + dt]``.

    PhreeqcRM does not advance its ``Time`` value itself.  Its documented
    control loop sets the current time to the beginning of the reaction
    interval and ``TimeStep`` to the integration length.  Keeping that
    distinction is essential for RATES definitions that use ``TOTAL_TIME``.
    """
    if not np.isfinite(start_time):
        raise CouplingError(
            f"Reaction start time must be finite: {start_time}"
        )
    if dt < 0.0 or not np.isfinite(dt):
        raise CouplingError(f"Reaction time step must be finite and nonnegative: {dt}")
    if not np.all(np.isfinite(transported)):
        raise CouplingError("Transport produced non-finite concentrations")
    if getattr(sim, "energy_enabled", False):
        from mf6pqc.energy import synchronize_temperature_to_chemistry

        synchronize_temperature_to_chemistry(sim)
    sim.phreeqc_rm.SetConcentrations(transported)
    sim.phreeqc_rm.SetTime(start_time * SECONDS_PER_DAY)
    sim.phreeqc_rm.SetTimeStep(dt * SECONDS_PER_DAY)
    sim.phreeqc_rm.RunCells()
    values = np.asarray(sim.phreeqc_rm.GetConcentrations(), dtype=float)
    if values.shape != reacted.shape:
        raise BackendError(
            f"PhreeqcRM returned concentration shape {values.shape}; expected {reacted.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise CouplingError("PhreeqcRM produced non-finite concentrations")
    reacted[:] = values


def get_calculated_density(sim) -> np.ndarray:
    """Return the configured chemistry density field in kg/m3."""
    if sim.use_phreeqc_calculated_density:
        density = np.asarray(sim.phreeqc_rm.GetDensityCalculated(), dtype=float)
    else:
        density = np.asarray(sim.selected_output[-1], dtype=float)
    density = density.ravel()
    if density.size != sim.nxyz:
        raise BackendError(
            f"Chemistry density has {density.size} cells; expected {sim.nxyz}"
        )
    density = density * DENSITY_SCALE
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise CouplingError("Chemistry produced non-positive or non-finite density")
    return density


def update_selected_output(sim) -> None:
    """Refresh and validate the selected-output matrix."""
    raw = np.asarray(sim.phreeqc_rm.GetSelectedOutput(), dtype=float)
    if raw.size % sim.nxyz != 0:
        raise BackendError(
            "Selected output size is not divisible by nxyz: "
            f"{raw.size} values for {sim.nxyz} cells"
        )
    sim.selected_output = raw.reshape(-1, sim.nxyz)
    if not np.all(np.isfinite(sim.selected_output)):
        raise CouplingError("PhreeqcRM selected output contains non-finite values")
    if sim.if_update_density and sim.use_phreeqc_calculated_density:
        sim.selected_output[-1] = get_calculated_density(sim) / DENSITY_SCALE


def synchronize_phreeqcrm_solution(
    sim,
    concentrations: np.ndarray,
    current_time: float,
    *,
    preserve_transport_endpoint: bool = False,
) -> None:
    """Re-speciate transported water without advancing kinetic reactions."""
    sim.phreeqc_rm.SetConcentrations(concentrations)
    if preserve_transport_endpoint:
        diagnostic_state = 2
        sim.phreeqc_rm.StateSave(diagnostic_state)
        try:
            sim.phreeqc_rm.SetTime(current_time * SECONDS_PER_DAY)
            sim.phreeqc_rm.SetTimeStep(0.0)
            sim.phreeqc_rm.RunCells()
            update_selected_output(sim)
        finally:
            sim.phreeqc_rm.StateApply(diagnostic_state)
            sim.phreeqc_rm.StateDelete(diagnostic_state)
        return
    sim.phreeqc_rm.SetTime(current_time * SECONDS_PER_DAY)
    sim.phreeqc_rm.SetTimeStep(0.0)
    sim.phreeqc_rm.RunCells()
    update_selected_output(sim)


def cache_source_variables(
    modflow_api, components: list[str], nxyz: int | None = None
) -> dict[str, dict[str, Any]]:
    """Cache live GWT SRC mass-rate arrays for chemical components."""
    variables: dict[str, dict[str, Any]] = {}
    for component in components:
        address = modflow_api.get_var_address(
            "SMASSRATE", get_gwt_model_name(component), "SRC"
        )
        pointer = modflow_api.get_value_ptr(address)
        if nxyz is not None and pointer.size != nxyz:
            raise BackendError(
                f"SRC for {component!r} has {pointer.size} cells; expected {nxyz}"
            )
        variables[component] = {"ptr": pointer}
    return variables


def update_water_only_sink_sources(sim, state: StandardCouplingState) -> None:
    """Cancel numerical solute export at configured pure-water sinks."""
    if state.water_sink_sources is None:
        return
    for component, concentration_info in state.concentration_variables.items():
        source_pointer = state.water_sink_sources[component]["ptr"]
        if component == "H2O":
            source_pointer[:] = 0.0
        else:
            concentration = np.asarray(concentration_info["ptr"]).reshape(-1)
            source_pointer[:] = sim.water_only_sink_rates * concentration


def _record_solver_failure(sim, solution_id: int, iterations: int, *, picard: bool) -> None:
    failure = {
        "time_days": float(sim.modflow_api.get_current_time()),
        "solution_id": int(solution_id),
        "iterations": int(iterations),
    }
    sim.modflow_convergence_failures.append(failure)
    context = "Picard solution" if picard else "solution"
    message = (
        f"MODFLOW 6 {context} failed to converge: solution={solution_id}, "
        f"time={failure['time_days']:.6g} days, iterations={iterations}"
    )
    if sim.fail_on_nonconvergence:
        raise ConvergenceError(message)
    print(f"Warning: {message}")


def solve_modflow_solutions(
    sim, state: StandardCouplingState, current_density: np.ndarray | None
) -> None:
    """Solve all registered MODFLOW solutions for the prepared time step."""
    for solution_id, iteration_pointer in state.solution_iterations.items():
        sim.modflow_api.prepare_solve(solution_id)
        if sim.if_update_density and solution_id == 1 and current_density is not None:
            sim.density_ptr[:] = current_density
        # prepare_solve reloads stress-period package arrays.
        update_water_only_sink_sources(sim, state)
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
                sim, solution_id, iterations, picard=False
            )


def should_save_time_step(sim, logical_step: int) -> bool:
    """Return whether a completed zero-based logical step is retained."""
    if sim.save_steps is not None:
        return (logical_step + 1) in sim.save_steps
    if sim.save_interval <= 0:
        raise ValueError("save_interval must be a positive integer")
    return (logical_step + sim.save_interval_offset) % sim.save_interval == 0


def should_run_reaction(sim, logical_step: int) -> bool:
    """Return whether chemistry is scheduled after this transport step."""
    if sim.reaction_steps is None:
        return True
    return (logical_step + 1) in sim.reaction_steps


def save_time_step_results(
    sim, logical_step: int, current_time: float | None = None
) -> None:
    """Store a copy of selected output when the output schedule requests it."""
    if should_save_time_step(sim, logical_step):
        sim.results.append(sim.selected_output.copy())
        if current_time is not None:
            sim.result_times.append(float(current_time))


def log_progress(
    current_time: float,
    end_time: float,
    completed_steps: int,
    suffix: str = "",
    interval: int = 1000,
) -> None:
    """Print progress after the first step and then every ``interval`` steps."""
    if completed_steps == 1 or completed_steps % interval == 0:
        extra = f", {suffix}" if suffix else ""
        print(
            f"  t = {current_time:.2f}/{end_time:.2f} days, "
            f"step={completed_steps}{extra}"
        )


def build_standard_state(sim) -> StandardCouplingState:
    """Create all runtime buffers and live pointers for SNIA or Strang."""
    concentration_variables = cache_concentration_variables(
        sim.modflow_api, sim.components, sim.nxyz
    )
    current_k11 = prepare_feedback(sim)
    solution_iterations = cache_solution_iterations(sim.modflow_api)
    transported, reacted = allocate_concentration_buffers(sim.nxyz, sim.ncomps)
    species_slices = build_species_slices(sim.nxyz, sim.ncomps)
    water_sink_sources = None
    if sim.has_water_only_sinks:
        water_sink_sources = cache_source_variables(
            sim.modflow_api, sim.components, sim.nxyz
        )
    time_step_schedule = build_time_step_schedule(sim.modflow_api)
    if sim.reaction_steps is not None:
        final_step = int(time_step_schedule.size)
        if max(sim.reaction_steps) > final_step:
            raise CouplingError(
                "reaction_steps contains a step beyond the MODFLOW TDIS schedule: "
                f"{max(sim.reaction_steps)} > {final_step}"
            )
        if final_step not in sim.reaction_steps:
            raise CouplingError(
                "reaction_steps must include the final MODFLOW transport step "
                f"({final_step})"
            )
        if sim.save_steps is not None and not sim.save_steps.issubset(
            sim.reaction_steps
        ):
            raise CouplingError(
                "save_steps must be a subset of reaction_steps because selected "
                "chemical output is refreshed only when chemistry runs"
            )

    current_time = float(sim.modflow_api.get_current_time())
    return StandardCouplingState(
        concentration_variables=concentration_variables,
        species_slices=species_slices,
        water_sink_sources=water_sink_sources,
        transported=transported,
        reacted=reacted,
        solution_iterations=solution_iterations,
        current_time=current_time,
        end_time=float(sim.modflow_api.get_end_time()),
        logical_step=0,
        current_k11=current_k11,
        time_step_schedule=time_step_schedule,
        last_reaction_time=current_time,
    )


def finalize_results(sim, logical_steps: int, start_wall_time: float) -> None:
    """Freeze output lists into arrays and report elapsed wall time."""
    sim.results = np.asarray(sim.results)
    sim.result_times = np.asarray(sim.result_times, dtype=float)
    if sim.result_times.size != sim.results.shape[0]:
        raise CouplingError(
            "Stored result times do not align with selected-output frames: "
            f"{sim.result_times.size} != {sim.results.shape[0]}"
        )
    if sim.if_update_porosity_K:
        sim.results_porosity = np.asarray(sim.results_porosity)
        sim.results_K = np.asarray(sim.results_K)
    if sim.if_update_diffc:
        sim.results_diffc = np.asarray(sim.results_diffc)
    if getattr(sim, "energy_enabled", False):
        from mf6pqc.energy import finalize_energy_results

        finalize_energy_results(sim)
    sim.final_time_step_index = logical_steps
    elapsed = time.perf_counter() - start_wall_time
    sim.last_run_wall_time_seconds = elapsed
    print(
        f"--- Simulation finished, steps={logical_steps}, time={elapsed:.2f} s ---"
    )


# Historical internal names retained for compatibility.
cache_src_var_info = cache_source_variables


def finalize_standard(sim, state: StandardCouplingState, start: float) -> None:
    """Compatibility wrapper for the pre-refactor finalization helper."""
    finalize_results(sim, state.logical_step, start)
