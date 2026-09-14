from __future__ import annotations

import numpy as np

from mf6pqc.constants import DENSITY_SCALE, MAX_POROSITY, MIN_POROSITY
from mf6pqc.exceptions import BackendError, CouplingError


def extract_output_information(
    headings: list[str], vm_minerals: dict[str, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve ``d_<mineral>`` selected-output rows and molar volumes.

    Molar volumes must be in L/mol.  With PhreeqcRM mineral amounts expressed
    per litre of representative volume, their product is a bulk-volume
    fraction and can therefore be subtracted from porosity.
    """
    indices: list[int] = []
    volumes: list[float] = []
    names: list[str] = []
    for index, heading in enumerate(headings):
        if not (heading.startswith("d_") and len(heading) > 2):
            continue
        mineral = heading[2:]
        if mineral not in vm_minerals:
            raise ValueError(
                f"No molar volume is configured for selected-output mineral {mineral!r}"
            )
        volume = float(vm_minerals[mineral])
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError(f"Molar volume for {mineral!r} must be finite and positive")
        indices.append(index)
        volumes.append(volume)
        names.append(mineral)
    return (
        np.asarray(indices, dtype=int),
        np.asarray(volumes, dtype=float).reshape(-1, 1),
        np.asarray(names, dtype=str),
    )


def calculate_porosity(
    selected_output: np.ndarray,
    output_indices: np.ndarray,
    mineral_volumes: np.ndarray,
    porosity: np.ndarray,
) -> np.ndarray:
    """Apply incremental mineral-volume changes to the porosity field."""
    selected = np.asarray(selected_output, dtype=float)
    current = np.asarray(porosity, dtype=float).ravel()
    indices = np.asarray(output_indices, dtype=int).ravel()
    volumes = np.asarray(mineral_volumes, dtype=float).reshape(-1, 1)
    if selected.ndim != 2 or selected.shape[1] != current.size:
        raise ValueError("selected_output must have shape (noutput, nxyz) matching porosity")
    if indices.size != volumes.shape[0]:
        raise ValueError("Mineral output indices and molar volumes have different lengths")
    if indices.size == 0:
        return current.copy()
    if np.any(indices < 0) or np.any(indices >= selected.shape[0]):
        raise IndexError("Mineral selected-output index is out of range")
    mineral_delta_moles = selected[indices, :]
    if not np.all(np.isfinite(mineral_delta_moles)):
        raise ValueError("Mineral mole changes contain non-finite values")
    total_volume_change = np.sum(volumes * mineral_delta_moles, axis=0)
    return current - total_volume_change


def update_diffc(new_porosity: np.ndarray, d0: np.ndarray) -> np.ndarray:
    """Return ``D_e = phi**(1/3) * D0`` for each cell."""
    porosity = np.asarray(new_porosity, dtype=float)
    free_water = np.asarray(d0, dtype=float)
    if porosity.shape != free_water.shape:
        raise ValueError(
            f"Porosity shape {porosity.shape} does not match d0 shape {free_water.shape}"
        )
    if (
        not np.all(np.isfinite(porosity))
        or not np.all(np.isfinite(free_water))
        or np.any(porosity < 0.0)
        or np.any(free_water < 0.0)
    ):
        raise ValueError("Porosity and d0 must be finite and nonnegative")
    return np.cbrt(porosity) * free_water


def get_calculated_density(sim) -> np.ndarray:
    """Return the configured chemistry density field in kg/m3."""
    if sim.use_phreeqc_calculated_density:
        density = np.asarray(sim.phreeqc_rm.GetDensityCalculated(), dtype=float)
    else:
        row = getattr(getattr(sim, "chemistry", None), "density_row", -1)
        density = np.asarray(sim.selected_output[row], dtype=float)
    density = density.ravel()
    if density.size != sim.nxyz:
        raise BackendError(f"Chemistry density has {density.size} cells; expected {sim.nxyz}")
    density = density * DENSITY_SCALE
    if not np.all(np.isfinite(density)) or np.any(density <= 0.0):
        raise CouplingError("Chemistry produced non-positive or non-finite density")
    return density


def update_porosity(selected_output, output_indices, mineral_volumes, porosity):
    return np.clip(
        calculate_porosity(selected_output, output_indices, mineral_volumes, porosity),
        MIN_POROSITY,
        MAX_POROSITY,
    )
