from __future__ import annotations

import logging
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from mf6pqc.exceptions import CouplingError

_logger = logging.getLogger(__name__)


class FrameBuffer:
    def __init__(self, capacity, shape, initial=(), *, storage="memory"):
        shape = (capacity, *shape)
        if storage == "disk" and capacity:
            with tempfile.TemporaryFile() as stream:
                stream.truncate(int(np.prod(shape)) * np.dtype(float).itemsize)
                self.values = np.memmap(stream, dtype=float, mode="r+", shape=shape)
        else:
            self.values = np.empty(shape, dtype=float)
        self.count = 0
        for values in initial:
            self.append(values)

    def append(self, values):
        if self.count >= self.values.shape[0]:
            raise CouplingError("Result history exceeded its configured save schedule")
        if np.shape(values) != self.values.shape[1:]:
            raise CouplingError("Result frame shape changed during the simulation")
        self.values[self.count] = values
        self.count += 1

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return self.values[: self.count][index]

    def __array__(self, dtype=None, copy=None):
        values = np.asarray(self.values[: self.count], dtype=dtype)
        return values.copy() if copy else values


@dataclass(slots=True)
class ResultHistory:
    results: object = field(default_factory=list)
    result_times: object = field(default_factory=list)
    results_porosity: object = field(default_factory=list)
    results_K: object = field(default_factory=list)
    results_diffc: object = field(default_factory=list)
    results_temperature: object = field(default_factory=list)
    results_temperature_for_flow: object = field(default_factory=list)
    results_viscosity: object = field(default_factory=list)
    results_reference_K: object = field(default_factory=list)
    results_effective_K: object = field(default_factory=list)

    def prepare(self, count, shapes, storage):
        for name, shape in shapes.items():
            capacity = count - 1 if name == "results_diffc" else count
            setattr(self, name, FrameBuffer(capacity, shape, getattr(self, name), storage=storage))


def should_save_time_step(sim, logical_step: int) -> bool:
    if sim.save_steps is not None:
        return (logical_step + 1) in sim.save_steps
    if sim.save_interval <= 0:
        raise ValueError("save_interval must be a positive integer")
    return (logical_step + sim.save_interval_offset) % sim.save_interval == 0


def prepare_results(sim, state, total_steps):
    if sim.save_steps is not None and max(sim.save_steps) > total_steps:
        raise CouplingError(f"save_steps exceeds the {total_steps} logical steps")
    history = getattr(sim, "history", None)
    if history is None:
        return
    if sim.reaction_steps is not None:
        saved = sum(should_save_time_step(sim, step - 1) for step in sim.reaction_steps)
    elif sim.save_steps is not None:
        saved = len(sim.save_steps)
    else:
        first = (-sim.save_interval_offset) % sim.save_interval
        saved = max(0, (total_steps - 1 - first) // sim.save_interval + 1)
    shapes = {"results": (len(sim.headings), sim.nxyz), "result_times": ()}
    if sim.if_update_porosity_K:
        shapes.update(results_porosity=(sim.nxyz,), results_K=(sim.nxyz,))
    if sim.if_update_diffc:
        shapes["results_diffc"] = (sim.nxyz,)
    if sim.energy_enabled:
        shapes.update(results_temperature=(sim.nxyz,), results_temperature_for_flow=(sim.nxyz,))
        if sim.vsc_enabled:
            shapes.update(
                results_viscosity=(sim.nxyz,),
                results_reference_K=(sim.nxyz,),
                results_effective_K=(sim.nxyz,),
            )
    history.prepare(saved + 1, shapes, sim.result_storage)


def append_frame(container, values):
    container.append(values if isinstance(container, FrameBuffer) else np.asarray(values).copy())


def save_time_step_results(sim, logical_step, current_time=None, *, current_k11=None):
    if not should_save_time_step(sim, logical_step):
        return
    append_frame(sim.results, sim.selected_output)
    if current_time is not None:
        sim.result_times.append(float(current_time))
    if sim.if_update_porosity_K:
        append_frame(sim.results_porosity, sim.porosity)
        append_frame(sim.results_K, current_k11)
    if sim.if_update_diffc:
        append_frame(sim.results_diffc, sim.current_diffusion)
    if getattr(sim, "energy_enabled", False):
        from mf6pqc.energy import save_energy_time_step_results

        save_energy_time_step_results(sim, logical_step)


class ProgressReporter:
    def __init__(self, end_time, total_steps, interval, *, clock=time.perf_counter):
        self.end_time = end_time
        self.total_steps = total_steps
        self.interval = max(1, min(interval, max(1, total_steps // 10)))
        self.clock = clock
        self.last_wall = clock()
        self.last_step = -1

    def report(self, state):
        completed = state.logical_step
        now = self.clock()
        due = completed <= 1 or completed >= self.total_steps or completed % self.interval == 0
        if completed == self.last_step or (not due and now - self.last_wall < 30.0):
            return
        percent = 100.0 * completed / max(1, self.total_steps)
        suffix = (
            f", SIA iters={state.picard_iteration + 1}"
            if hasattr(state, "picard_iteration") and completed
            else ""
        )
        _logger.info(
            "  t = %.6g/%.6g days, step=%d/%d (%.1f%%)%s",
            state.current_time,
            self.end_time,
            completed,
            self.total_steps,
            percent,
            suffix,
        )
        self.last_step = completed
        self.last_wall = now
