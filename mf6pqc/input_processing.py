"""Validated translation of chemical initial conditions to PhreeqcRM."""

from __future__ import annotations

from collections.abc import Mapping
import numbers
import warnings

import numpy as np

from mf6pqc.constants import IC_DEFAULT, MODULE_INDICES
from mf6pqc.types import ArrayLike
from mf6pqc.utils import ensure_array


def _condition_indices(nxyz: int, module_name: str, value) -> np.ndarray:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        indices = np.full(nxyz, int(value), dtype=np.int32)
    elif isinstance(value, numbers.Real) and float(value).is_integer():
        indices = np.full(nxyz, int(value), dtype=np.int32)
    elif isinstance(value, (list, tuple, np.ndarray)):
        raw = np.asarray(value)
        if raw.size != nxyz:
            raise ValueError(
                f"Initial-condition array for {module_name!r} has {raw.size} "
                f"values; expected {nxyz}"
            )
        numeric = np.asarray(raw, dtype=float).ravel()
        if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
            raise ValueError(
                f"Initial-condition indices for {module_name!r} must be finite integers"
            )
        if np.any(numeric > np.iinfo(np.int32).max):
            raise OverflowError(
                f"Initial-condition index for {module_name!r} exceeds int32"
            )
        indices = numeric.astype(np.int32)
    else:
        raise TypeError(
            f"Initial condition for {module_name!r} must be an integer or cell array"
        )
    if np.any(indices < IC_DEFAULT):
        raise ValueError(
            f"Initial-condition indices for {module_name!r} must be -1 or nonnegative"
        )
    return indices


def create_ic_array_from_map(
    nxyz: int, ic_map: Mapping[str, object], *, strict: bool = True
) -> np.ndarray:
    """Pack named PHREEQC entities in the seven-block PhreeqcRM order."""
    if not isinstance(ic_map, Mapping) or not ic_map:
        raise TypeError("ic_map must be a non-empty mapping")
    packed = np.full(nxyz * len(MODULE_INDICES), IC_DEFAULT, dtype=np.int32)
    for module_name, value in ic_map.items():
        if module_name not in MODULE_INDICES:
            message = (
                f"Unknown chemical module {module_name!r}; expected one of "
                f"{', '.join(MODULE_INDICES)}"
            )
            if strict:
                raise KeyError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            continue
        indices = _condition_indices(nxyz, module_name, value)
        block = MODULE_INDICES[module_name]
        packed[block * nxyz : (block + 1) * nxyz] = indices
    return packed


def setup_single_ic(phreeqc_rm, nxyz: int, ic_map: Mapping[str, object]) -> None:
    """Apply one initial-condition mapping to PhreeqcRM."""
    print("--- Setting single initial chemical condition ---")
    phreeqc_rm.InitialPhreeqc2Module(create_ic_array_from_map(nxyz, ic_map))


def setup_mixed_ic(
    phreeqc_rm,
    nxyz: int,
    ic_map1: Mapping[str, object],
    ic_map2: Mapping[str, object],
    fractions: ArrayLike,
) -> None:
    """Mix two initial-condition mappings cell by cell."""
    print("--- Setting mixed initial chemical condition ---")
    first = create_ic_array_from_map(nxyz, ic_map1)
    second = create_ic_array_from_map(nxyz, ic_map2)
    fraction_array = ensure_array(nxyz, "mixing fractions", fractions)
    if (
        not np.all(np.isfinite(fraction_array))
        or np.any(fraction_array < 0.0)
        or np.any(fraction_array > 1.0)
    ):
        raise ValueError("mixing fractions must be finite and in [0, 1]")
    phreeqc_rm.InitialPhreeqc2Module_mix(
        first, second, np.tile(fraction_array, len(MODULE_INDICES))
    )
