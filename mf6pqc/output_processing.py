"""Selected-output interpretation and durable result serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from mf6pqc.constants import MAX_POROSITY, MIN_POROSITY


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


def update_porosity(
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
        raise ValueError(
            "selected_output must have shape (noutput, nxyz) matching porosity"
        )
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
    return np.clip(current - total_volume_change, MIN_POROSITY, MAX_POROSITY)


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


def _atomic_save(path: Path, values: Any) -> None:
    """Write one NumPy array atomically within its destination directory."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, np.asarray(values))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_results(
    output_dir: str,
    case_name: str,
    headings: list[str],
    results: np.ndarray,
    results_porosity: list | np.ndarray,
    results_k: list | np.ndarray,
    results_diffc: list | np.ndarray,
    if_update_porosity_k: bool,
    if_update_diffc: bool,
    filename: str | None = None,
    *,
    result_times: list[float] | np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
    energy_results: dict[str, list | np.ndarray] | None = None,
) -> None:
    """Validate and atomically save MF6PQC arrays plus a small manifest."""
    if not headings:
        raise ValueError("Cannot save results without selected-output headings")
    values = np.asarray(results)
    if values.ndim != 3:
        raise ValueError(
            f"results must have shape (time, output, cell); got {values.shape}"
        )
    if values.shape[1] != len(headings):
        raise ValueError(
            f"results contain {values.shape[1]} outputs but there are {len(headings)} headings"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("results contain non-finite values")
    times = None
    if result_times is not None:
        times = np.asarray(result_times, dtype=float).ravel()
        if times.size != values.shape[0]:
            raise ValueError(
                f"result_times has {times.size} entries for {values.shape[0]} result frames"
            )
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0.0):
            raise ValueError("result_times must be finite and nondecreasing")
    porosity_values = conductivity_values = diffusion_values = None
    if if_update_porosity_k:
        porosity_values = np.asarray(results_porosity, dtype=float)
        conductivity_values = np.asarray(results_k, dtype=float)
        expected_shape = (values.shape[0], values.shape[2])
        if porosity_values.shape != expected_shape:
            raise ValueError(
                f"Porosity results have shape {porosity_values.shape}; expected {expected_shape}"
            )
        if conductivity_values.shape != expected_shape:
            raise ValueError(
                f"K results have shape {conductivity_values.shape}; expected {expected_shape}"
            )
        if (
            not np.all(np.isfinite(porosity_values))
            or not np.all(np.isfinite(conductivity_values))
            or np.any(conductivity_values <= 0.0)
        ):
            raise ValueError("Porosity/K results contain invalid values")
    if if_update_diffc:
        diffusion_values = np.asarray(results_diffc, dtype=float)
        expected_shape = (max(0, values.shape[0] - 1), values.shape[2])
        if diffusion_values.shape != expected_shape:
            raise ValueError(
                f"Diffusion results have shape {diffusion_values.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(diffusion_values)) or np.any(diffusion_values < 0.0):
            raise ValueError("Diffusion results contain invalid values")

    thermal_values: dict[str, np.ndarray] = {}
    if energy_results is not None:
        allowed = {
            "temperature",
            "temperature_for_flow",
            "viscosity",
            "reference_K",
            "effective_K",
        }
        unknown = set(energy_results) - allowed
        if unknown:
            raise ValueError(f"Unknown energy result fields: {sorted(unknown)}")
        required = {"temperature", "temperature_for_flow"}
        missing = required - set(energy_results)
        if missing:
            raise ValueError(f"Missing energy result fields: {sorted(missing)}")
        expected_shape = (values.shape[0], values.shape[2])
        for name, raw in energy_results.items():
            field = np.asarray(raw, dtype=float)
            if field.shape != expected_shape:
                raise ValueError(
                    f"Energy field {name!r} has shape {field.shape}; "
                    f"expected {expected_shape}"
                )
            if not np.all(np.isfinite(field)):
                raise ValueError(f"Energy field {name!r} contains non-finite values")
            if name.startswith("temperature"):
                if np.any(field <= -273.15):
                    raise ValueError(f"Energy field {name!r} contains invalid temperatures")
            elif np.any(field <= 0.0):
                raise ValueError(f"Energy field {name!r} must be positive")
            thermal_values[name] = field

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result_path = destination / (filename or "results.npy")
    if result_path.suffix.lower() != ".npy":
        result_path = result_path.with_suffix(".npy")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    base = result_path.with_suffix("")
    _atomic_save(result_path, values)
    print(f"Results saved to: {result_path}")

    headings_path = Path(f"{base}_headings.txt")
    _atomic_write_text(headings_path, "".join(f"{heading}\n" for heading in headings))
    print(f"Headings saved to: {headings_path}")

    saved_files = [result_path.name, headings_path.name]
    if times is not None:
        times_path = Path(f"{base}_times.npy")
        _atomic_save(times_path, times)
        saved_files.append(times_path.name)

    if if_update_porosity_k:
        porosity_path = Path(f"{base}_porosity.npy")
        conductivity_path = Path(f"{base}_K.npy")
        _atomic_save(porosity_path, porosity_values)
        _atomic_save(conductivity_path, conductivity_values)
        saved_files.extend([porosity_path.name, conductivity_path.name])
        print(f"Porosity results saved to: {porosity_path}")
        print(f"K results saved to: {conductivity_path}")
    if if_update_diffc:
        diffusion_path = Path(f"{base}_diffc.npy")
        _atomic_save(diffusion_path, diffusion_values)
        saved_files.append(diffusion_path.name)
        print(f"DIFFC results saved to: {diffusion_path}")
    energy_files: dict[str, str] = {}
    for name, field in thermal_values.items():
        energy_path = Path(f"{base}_{name}.npy")
        _atomic_save(energy_path, field)
        saved_files.append(energy_path.name)
        energy_files[name] = energy_path.name
    if energy_files:
        print(f"Thermal/VSC results saved to: {base}_*.npy")

    manifest = {
        "schema_version": 1,
        "case_name": case_name,
        "result_shape": list(values.shape),
        "headings": list(headings),
        "files": saved_files,
        "has_porosity_and_k": bool(if_update_porosity_k),
        "has_diffusion": bool(if_update_diffc),
    }
    if thermal_values:
        manifest["has_energy"] = True
        manifest["energy"] = {
            "files": energy_files,
            "temperature_units": "degC",
            "viscosity_units": "MODFLOW VSC input units",
            "conductivity_semantics": {
                "reference_K": "NPF K11INPUT used by the completed flow solve",
                "effective_K": "viscosity-adjusted NPF K11 used by the completed flow solve",
            },
            "explicit_coupling": (
                "temperature_for_flow is the lagged GWE field used by VSC; "
                "temperature is the post-GWE field used by PhreeqcRM reactions"
            ),
        }
    if metadata:
        manifest["run"] = metadata
    manifest_path = Path(f"{base}_manifest.json")
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
