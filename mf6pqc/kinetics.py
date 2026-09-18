"""Declarative mineral kinetics for the implicit transport--reaction solver.

Amounts are mol/L bulk, aqueous concentrations are mol/L water, and time is
days. Positive rates dissolve a mineral. PHREEQC still supplies speciation;
these definitions, rather than PHREEQC RATES, own integration in Implicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np

from mf6pqc.exceptions import ConfigurationError
from mf6pqc.types import ArrayLike
from mf6pqc.utils import require_integer


@dataclass(frozen=True)
class KineticState:
    """Cell-local, read-only inputs to a custom kinetic driving force.

    Callbacks must return one value per cell, without modifying inputs or sharing
    state between cells. Use ``minerals`` for trial amounts, not USER_PUNCH KIN:
    native mineral amounts are committed only after Newton has converged.
    """

    concentrations: Mapping[str, np.ndarray]
    selected: Mapping[str, np.ndarray]
    minerals: Mapping[str, np.ndarray]
    temperature_c: np.ndarray
    time_days: float


@dataclass(frozen=True)
class KineticReaction:
    """A mineral with rate = k * (m / m_ref)**p * driving_force.

    ``stoichiometry`` gives transported component moles released per mole of
    mineral (including total H/O and signed Charge). ``rate_constant`` has units
    mol/L bulk/day and may vary by cell. ``surface_exponent`` is in [0, 1).
    The default driving force is 1 - SR, read from ``saturation_heading``.
    Alternatively a cell-local callback may use any selected chemical output,
    aqueous concentration, temperature, time, or trial mineral amount.

    ``minimum_amount`` (mol/L bulk) limits dissolution and preserves a seed
    for subsequent growth. The default is zero. For p > 0, a mineral at exactly
    zero remains exhausted (zero reactive surface).
    For p = 0, precipitation from zero is allowed. Reference amounts default to
    the initial inventory; specify them explicitly for initially empty cells.
    """

    name: str
    stoichiometry: Mapping[str, float]
    rate_constant: ArrayLike
    surface_exponent: float = 0.0
    reference_amount: ArrayLike | None = None
    saturation_heading: str | None = None
    driving_force: Callable[[KineticState], np.ndarray] | None = None
    saturation_index_heading: str | None = None
    minimum_amount: ArrayLike = 0.0

    def validated(self):
        if not isinstance(self.name, str) or not self.name or any(c.isspace() for c in self.name):
            raise ConfigurationError("KineticReaction.name must be one PHREEQC kinetic name")
        if (
            not self.stoichiometry
            or any(
                not isinstance(k, str) or not k or not np.isfinite(v)
                for k, v in self.stoichiometry.items()
            )
            or not any(self.stoichiometry.values())
        ):
            raise ConfigurationError(
                "Kinetic stoichiometry must contain finite, nonzero coefficients"
            )
        if not np.isfinite(self.surface_exponent) or not 0 <= self.surface_exponent < 1:
            raise ConfigurationError("surface_exponent must be in [0, 1)")
        if self.driving_force is not None and not callable(self.driving_force):
            raise ConfigurationError("driving_force must be callable or None")
        for name in ("rate_constant", "reference_amount", "minimum_amount"):
            value = getattr(self, name)
            if value is None and name != "reference_amount":
                raise ConfigurationError(f"{name} must be finite and nonnegative")
            if value is not None and (
                not np.all(np.isfinite(value)) or np.any(np.asarray(value) < 0)
            ):
                raise ConfigurationError(f"{name} must be finite and nonnegative")
        for heading in (self.saturation_heading, self.saturation_index_heading):
            if heading is not None and (not isinstance(heading, str) or not heading.strip()):
                raise ConfigurationError("Saturation headings must be nonempty strings")
        return replace(self, stoichiometry=dict(self.stoichiometry))


@dataclass(slots=True)
class ImplicitOptions:
    """Controls for transformed backward Euler with frozen medium properties.

    Newton tolerances control algebraic error, not time-discretization error.
    Validate the TDIS schedule by refinement against the quantities of interest.
    """

    reactions: tuple[KineticReaction, ...] = ()
    maximum_iterations: int = 50
    absolute_tolerance: float = 1.0e-8
    relative_tolerance: float = 0.0
    concentration_tolerance: float = 1.0e-7
    derivative_step: float = 1.0e-8
    derivative_refresh: int = 8
    dense_limit: int = 400
    predict_porosity: bool = True
    chemical_jacobian: str = "finite_difference"
    directed_transport_blocks: bool = True
    porosity_coupling: str = "conservative"

    def validated(self):
        options = replace(self)
        if options.chemical_jacobian not in {"finite_difference", "species"}:
            raise ConfigurationError("chemical_jacobian must be finite_difference or species")
        if options.porosity_coupling not in {"conservative", "lagged"}:
            raise ConfigurationError("porosity_coupling must be conservative or lagged")
        options.reactions = tuple(r.validated() for r in self.reactions)
        names = [r.name.casefold() for r in options.reactions]
        if len(set(names)) != len(names):
            raise ConfigurationError("Implicit kinetic names must be unique")
        for name in ("maximum_iterations", "derivative_refresh", "dense_limit"):
            setattr(options, name, require_integer(name, getattr(options, name)))
        for name in ("absolute_tolerance", "concentration_tolerance", "derivative_step"):
            if not np.isfinite(getattr(options, name)) or getattr(options, name) <= 0:
                raise ConfigurationError(f"Implicit {name} must be finite and positive")
        if not np.isfinite(options.relative_tolerance) or options.relative_tolerance < 0:
            raise ConfigurationError("Implicit relative_tolerance must be finite and nonnegative")
        return options


__all__ = ["ImplicitOptions", "KineticReaction", "KineticState"]
