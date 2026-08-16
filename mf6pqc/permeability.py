"""Replaceable constitutive models for MODFLOW hydraulic conductivity.

Despite the historical module name, MODFLOW NPF ``K`` is hydraulic
conductivity (length/time), not intrinsic permeability (length squared).
MF6PQC keeps the historical class names for compatibility, but every updater
in this module returns hydraulic conductivity values suitable for NPF.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np


def _field(value, name: str, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


def _base_fields(
    hydraulic_conductivity_old,
    porosity_old,
    porosity_new,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conductivity = np.asarray(hydraulic_conductivity_old, dtype=float)
    if conductivity.ndim != 1:
        conductivity = conductivity.ravel()
    if not np.all(np.isfinite(conductivity)) or np.any(conductivity <= 0.0):
        raise ValueError("hydraulic_conductivity_old must be finite and positive")
    old = _field(porosity_old, "porosity_old", conductivity.shape)
    new = _field(porosity_new, "porosity_new", conductivity.shape)
    if np.any(old <= 0.0) or np.any(old > 1.0):
        raise ValueError("porosity_old must be in (0, 1]")
    if np.any(new <= 0.0) or np.any(new > 1.0):
        raise ValueError("porosity_new must be in (0, 1]")
    return conductivity, old, new


def _kozeny_carman_factor(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    # The relationship is singular at phi=1.  Clipping only protects the
    # numerical evaluation; MF6PQC separately records the clipped porosity.
    epsilon = 1.0e-12
    old_eval = np.clip(old, epsilon, 1.0 - epsilon)
    new_eval = np.clip(new, epsilon, 1.0 - epsilon)
    old_term = old_eval**3 / (1.0 - old_eval) ** 2
    new_term = new_eval**3 / (1.0 - new_eval) ** 2
    return new_term / old_term


class BasePermeabilityUpdater(ABC):
    """Historical interface for a cell-wise NPF K update model."""

    @abstractmethod
    def update(
        self,
        K_old: np.ndarray,
        porosity_old: np.ndarray,
        porosity_new: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the updated MODFLOW hydraulic-conductivity field."""
        raise NotImplementedError


# Scientifically precise spelling for new code; the old name remains public.
BaseHydraulicConductivityUpdater = BasePermeabilityUpdater


class KozenyCarmanUpdater(BasePermeabilityUpdater):
    """Scale NPF K by the ratio of Kozeny-Carman porosity factors.

    ``K_new = K_old * f(phi_new) / f(phi_old)`` where
    ``f(phi) = phi**3 / (1 - phi)**2``.
    """

    def update(
        self,
        K_old: np.ndarray,
        porosity_old: np.ndarray,
        porosity_new: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        conductivity, old, new = _base_fields(K_old, porosity_old, porosity_new)
        return conductivity * _kozeny_carman_factor(old, new)


class FluidAdjustedKozenyCarmanUpdater(BasePermeabilityUpdater):
    """Kozeny-Carman K update with explicit fluid-property scaling.

    The additional factor is ``rho_new/rho_old * mu_old/mu_new``.  This is an
    opt-in model: do not use it when MODFLOW BUY and VSC already account for
    density and viscosity, because doing so may count fluid effects twice.
    """

    def update(
        self,
        K_old: np.ndarray,
        porosity_old: np.ndarray,
        porosity_new: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        conductivity, old, new = _base_fields(K_old, porosity_old, porosity_new)
        shape = conductivity.shape
        density_old_values = (
            np.ones(shape) if density_old is None else _field(density_old, "density_old", shape)
        )
        density_new_values = (
            np.ones(shape) if density_new is None else _field(density_new, "density_new", shape)
        )
        viscosity_old_values = (
            np.ones(shape)
            if viscosity_old is None
            else _field(viscosity_old, "viscosity_old", shape)
        )
        viscosity_new_values = (
            np.ones(shape)
            if viscosity_new is None
            else _field(viscosity_new, "viscosity_new", shape)
        )
        if (
            np.any(density_old_values <= 0.0)
            or np.any(density_new_values <= 0.0)
            or np.any(viscosity_old_values <= 0.0)
            or np.any(viscosity_new_values <= 0.0)
        ):
            raise ValueError("density and viscosity fields must be positive")
        return (
            conductivity
            * _kozeny_carman_factor(old, new)
            * density_new_values
            / density_old_values
            * viscosity_old_values
            / viscosity_new_values
        )


# Compatibility name used by the earlier prototype.
DensityCoupledKozenyCarmanUpdater = FluidAdjustedKozenyCarmanUpdater


class PowerLawUpdater(BasePermeabilityUpdater):
    """Scale NPF K with ``(phi_new / phi_old)**exponent``."""

    def __init__(self, n: float = 2.0):
        if not np.isfinite(n):
            raise ValueError("Power-law exponent must be finite")
        self.n = float(n)

    def update(
        self,
        K_old: np.ndarray,
        porosity_old: np.ndarray,
        porosity_new: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        conductivity, old, new = _base_fields(K_old, porosity_old, porosity_new)
        return conductivity * np.power(new / old, self.n)


class MLPermeabilityUpdater(BasePermeabilityUpdater):
    """Adapter for a trained model that predicts an NPF K field.

    The default preserves the historical input contract and passes the new
    porosity vector to ``model.predict``.  A ``feature_builder`` can later add
    mineral, density, temperature, or history features without coupling the
    core simulator to a specific machine-learning library.
    """

    def __init__(
        self,
        model: Any,
        feature_builder: Callable[..., Any] | None = None,
    ) -> None:
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model must expose a callable predict method")
        self.model = model
        self.feature_builder = feature_builder

    def update(
        self,
        K_old: np.ndarray,
        porosity_old: np.ndarray,
        porosity_new: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        conductivity, old, new = _base_fields(K_old, porosity_old, porosity_new)
        if self.feature_builder is None:
            features = new
        else:
            features = self.feature_builder(
                K_old=conductivity,
                porosity_old=old,
                porosity_new=new,
                density_old=density_old,
                density_new=density_new,
                viscosity_old=viscosity_old,
                viscosity_new=viscosity_new,
            )
        prediction = np.asarray(self.model.predict(features), dtype=float).ravel()
        if prediction.shape != conductivity.shape:
            raise ValueError(
                f"ML model returned shape {prediction.shape}; expected {conductivity.shape}"
            )
        if not np.all(np.isfinite(prediction)) or np.any(prediction <= 0.0):
            raise ValueError("ML model returned non-positive or non-finite K values")
        return prediction


__all__ = [
    "BaseHydraulicConductivityUpdater",
    "BasePermeabilityUpdater",
    "DensityCoupledKozenyCarmanUpdater",
    "FluidAdjustedKozenyCarmanUpdater",
    "KozenyCarmanUpdater",
    "MLPermeabilityUpdater",
    "PowerLawUpdater",
]
