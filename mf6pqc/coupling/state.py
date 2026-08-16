"""Typed runtime state owned by coupling algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
