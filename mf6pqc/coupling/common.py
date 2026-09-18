"""Shared primitives for all MF6PQC coupling algorithms."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from mf6pqc.backends import CheckedPhreeqcRM, solve_prepared_modflow
from mf6pqc.constants import (
    MIN_CONCENTRATION,
    SECONDS_PER_DAY,
)
from mf6pqc.exceptions import BackendError, ConvergenceError, CouplingError
from mf6pqc.feedback import prepare_feedback
from mf6pqc.properties import get_calculated_density as get_calculated_density
from mf6pqc.results import (
    ProgressReporter,
    prepare_results,
)
from mf6pqc.results import (
    save_time_step_results as save_time_step_results,
)
from mf6pqc.results import (
    should_save_time_step as should_save_time_step,
)
from mf6pqc.utils import get_gwt_model_name, get_species_slice

_logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StandardCouplingState:
    """Mutable state for SNIA and Strang coupling.

    Concentration arrays use PhreeqcRM's component-major layout: all cells for
    component 0, followed by all cells for component 1, and so on.
    """

    concentration_variables: dict[str, dict[str, Any]]
    species_slices: tuple[slice, ...]
    water_sink_sources: dict[str, dict[str, Any]] | None
    transported: np.ndarray
    reacted: np.ndarray
    solution_iterations: dict[int, np.ndarray]
    current_time: float
    end_time: float
    logical_step: int
    current_k11: np.ndarray | None
    time_step_schedule: np.ndarray
    transport_step: int = 0
    last_reaction_time: float = 0.0
    nonnegative_slices: tuple[slice, ...] | None = None


@dataclass(slots=True)
class SIACouplingState:
    """Mutable state for source-based sequential iterative coupling."""

    concentration_variables: dict[str, dict[str, Any]]
    species_slices: tuple[slice, ...]
    transported: np.ndarray
    reaction_input: np.ndarray
    reacted: np.ndarray
    previous_time_concentrations: np.ndarray
    previous_iteration_concentrations: np.ndarray
    source_rates: np.ndarray
    candidate_source_rates: np.ndarray
    concentration_difference: np.ndarray
    coupling_difference: np.ndarray
    source_difference: np.ndarray
    solution_iterations: dict[int, np.ndarray]
    source_variables: dict[str, dict[str, Any]]
    bulk_cell_volume: np.ndarray
    mobile_water_volume: np.ndarray
    current_time: float
    end_time: float
    logical_step: int
    current_k11: np.ndarray | None
    previous_density: np.ndarray | None
    candidate_density: np.ndarray | None
    density_difference: np.ndarray | None
    picard_iteration: int
    time_step_schedule: np.ndarray
    current_dt: float = 0.0
    nonnegative_slices: tuple[slice, ...] | None = None


@dataclass(frozen=True, slots=True)
class CouplingHooks:
    """Observe live solver state; copy arrays when retaining a snapshot.

    Transport and reaction hooks run before SNIA commits chemistry feedback.
    Step hooks run after a complete logical step for every coupling method.
    """

    on_initialize: Callable[[Any, Any], None] | None = None
    on_transport: Callable[[Any, StandardCouplingState, float], None] | None = None
    on_reaction: Callable[[Any, StandardCouplingState], None] | None = None
    on_step: Callable[[Any, Any], None] | None = None

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if value is not None and not callable(value):
                raise TypeError(f"{item.name} must be callable or None")


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
    if not np.all(np.isfinite(sim.cell_thick)) or np.any(sim.cell_thick <= 0.0):
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
        _logger.info(f"  - solute '{component}', shape={pointer.shape}")
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
            raise BackendError(f"Invalid MXITER for MODFLOW solution {solution_id}: {pointer!r}")
        iterations[solution_id] = pointer
    return iterations


def allocate_concentration_buffers(nxyz: int, ncomps: int) -> tuple[np.ndarray, np.ndarray]:
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
    """Expand retained MODFLOW TDIS period data into explicit step lengths."""
    # Read the runtime copies: MODFLOW 6.8 releases the __INPUT__ arrays
    # during initialization. TDIS retains the full arrays for all periods.
    perlen = np.asarray(
        modflow_api.get_value(modflow_api.get_var_address("PERLEN", "TDIS")),
        dtype=float,
    ).ravel()
    nstp = np.asarray(
        modflow_api.get_value(modflow_api.get_var_address("NSTP", "TDIS")),
        dtype=float,
    ).ravel()
    tsmult = np.asarray(
        modflow_api.get_value(modflow_api.get_var_address("TSMULT", "TDIS")),
        dtype=float,
    ).ravel()
    if not (perlen.size == nstp.size == tsmult.size):
        raise BackendError("TDIS PERLEN, NSTP, and TSMULT lengths are inconsistent")
    if perlen.size == 0:
        raise BackendError("TDIS contains no stress periods")

    periods: list[np.ndarray] = []
    for period_length, step_count_raw, multiplier in zip(perlen, nstp, tsmult, strict=False):
        if not np.isfinite(step_count_raw) or step_count_raw != np.floor(step_count_raw):
            raise BackendError("TDIS NSTP must contain finite integer step counts")
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
        if multiplier == 1.0:
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


def advance_to_end(sim, state, step, *, total_steps=None) -> None:
    total_steps = state.time_step_schedule.size if total_steps is None else total_steps
    prepare_results(sim, state, total_steps)
    hooks = getattr(sim, "_coupling_hooks", None)
    if hooks is not None and hooks.on_initialize is not None:
        hooks.on_initialize(sim, state)
    on_step = None if hooks is None else hooks.on_step
    progress = ProgressReporter(state.end_time, total_steps, sim.progress_interval)
    progress.report(state)
    limit = state.end_time - time_tolerance(state.end_time)
    while state.current_time < limit:
        step(sim, state)
        if on_step is not None:
            on_step(sim, state)
        progress.report(state)


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


def read_concentrations_from_phreeqcrm(sim, destination: np.ndarray) -> None:
    """Read and validate concentrations in the backend's current volume basis."""
    values = np.asarray(sim.phreeqc_rm.GetConcentrations(), dtype=float)
    if values.shape != destination.shape:
        raise BackendError(
            f"PhreeqcRM returned concentration shape {values.shape}; expected {destination.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise CouplingError("PhreeqcRM produced non-finite concentrations")
    if values is not destination:
        destination[:] = values


def commit_reaction_concentrations(sim, state: StandardCouplingState | SIACouplingState) -> None:
    """Write the reaction endpoint after committing medium-property feedback.

    With UseSolutionDensityVolume(False), GetConcentrations divides the stored
    aqueous moles by the current porosity * saturation * representative volume.
    SetPorosity changes that denominator, so the pre-feedback reaction buffer
    must be refreshed before GWT uses the new MST porosity. Let PhreeqcRM apply
    its configured volume convention instead of rescaling concentrations here.

    Reading concentrations neither advances reactions nor replaces selected
    output (including the mineral increments used for porosity feedback).
    Fixed-porosity runs keep the original buffer and make no extra backend call.
    """
    if sim.if_update_porosity_K:
        read_concentrations_from_phreeqcrm(sim, state.reacted)
    write_concentrations_to_modflow(
        state.concentration_variables, state.species_slices, state.reacted
    )


def enforce_component_domains(
    concentrations: np.ndarray,
    components: list[str] | tuple[str, ...],
    species_slices: tuple[slice, ...] | list[slice],
    signed_components: frozenset[str] | set[str] | tuple[str, ...] = ("charge",),
    *,
    nonnegative_slices=None,
) -> None:
    """Apply PHREEQC component domains after a transport solve.

    Element concentrations cannot be negative.  The ``Charge`` component is
    different: it stores charge imbalance in equivalents and its valid domain
    explicitly includes negative values.  Clipping it with the element totals
    changes alkalinity and can strongly perturb pH in otherwise dilute models.
    """
    if nonnegative_slices is None:
        nonnegative_slices = build_nonnegative_slices(components, species_slices, signed_components)
    for component_slice in nonnegative_slices:
        values = concentrations[component_slice]
        np.maximum(values, MIN_CONCENTRATION, out=values)


def build_nonnegative_slices(components, species_slices, signed_components):
    if len(components) != len(species_slices):
        raise ValueError("components and species_slices must have equal length")
    signed = {component.casefold() for component in signed_components}
    blocks = []
    for component, block in zip(components, species_slices, strict=True):
        if component.casefold() in signed:
            continue
        if blocks and blocks[-1].stop == block.start:
            blocks[-1] = slice(blocks[-1].start, block.stop)
        else:
            blocks.append(block)
    return tuple(blocks)


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
        raise CouplingError(f"Reaction start time must be finite: {start_time}")
    if dt < 0.0 or not np.isfinite(dt):
        raise CouplingError(f"Reaction time step must be finite and nonnegative: {dt}")
    if not np.all(np.isfinite(transported)):
        raise CouplingError("Transport produced non-finite concentrations")
    batched = isinstance(sim.phreeqc_rm, CheckedPhreeqcRM)
    temperature = None
    if getattr(sim, "energy_enabled", False):
        from mf6pqc.energy import synchronize_temperature_to_chemistry

        temperature = synchronize_temperature_to_chemistry(sim, write=not batched)
        if not sim.sync_gwe_temperature_to_phreeqc:
            temperature = None
    if batched:
        sim.phreeqc_rm.advance_into(
            transported,
            start_time * SECONDS_PER_DAY,
            dt * SECONDS_PER_DAY,
            reacted,
            sim.selected_output,
            temperature,
        )
    else:
        sim.phreeqc_rm.SetConcentrations(transported)
        sim.phreeqc_rm.SetTime(start_time * SECONDS_PER_DAY)
        sim.phreeqc_rm.SetTimeStep(dt * SECONDS_PER_DAY)
        sim.phreeqc_rm.RunCells()
    read_concentrations_from_phreeqcrm(sim, reacted)


def update_selected_output(sim) -> None:
    """Refresh and validate the selected-output matrix."""
    raw = np.asarray(sim.phreeqc_rm.GetSelectedOutput(), dtype=float)
    expected = len(sim.headings) * sim.nxyz
    if raw.size != expected:
        raise BackendError(f"Selected output size changed: {raw.size} values; expected {expected}")
    sim.selected_output = raw.reshape(-1, sim.nxyz)
    if not np.all(np.isfinite(sim.selected_output)):
        raise CouplingError("PhreeqcRM selected output contains non-finite values")


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
        address = modflow_api.get_var_address("SMASSRATE", get_gwt_model_name(component), "SRC")
        pointer = modflow_api.get_value_ptr(address)
        if nxyz is not None and pointer.size != nxyz:
            raise BackendError(f"SRC for {component!r} has {pointer.size} cells; expected {nxyz}")
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
    _logger.warning(message)


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
        converged, iterations = solve_prepared_modflow(sim.modflow_api, solution_id, maximum)
        if not converged:
            _record_solver_failure(sim, solution_id, iterations, picard=False)


def should_run_reaction(sim, logical_step: int) -> bool:
    """Return whether chemistry is scheduled after this transport step."""
    if sim.reaction_steps is None:
        return True
    return (logical_step + 1) in sim.reaction_steps


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
        water_sink_sources = cache_source_variables(sim.modflow_api, sim.components, sim.nxyz)
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
                f"reaction_steps must include the final MODFLOW transport step ({final_step})"
            )
        if sim.save_steps is not None and not sim.save_steps.issubset(sim.reaction_steps):
            raise CouplingError(
                "save_steps must be a subset of reaction_steps because selected "
                "chemical output is refreshed only when chemistry runs"
            )

    current_time = float(sim.modflow_api.get_current_time())
    return StandardCouplingState(
        concentration_variables=concentration_variables,
        species_slices=species_slices,
        nonnegative_slices=build_nonnegative_slices(
            sim.components, species_slices, sim.signed_components
        ),
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
    _logger.info(f"--- Simulation finished, steps={logical_steps}, time={elapsed:.2f} s ---")
