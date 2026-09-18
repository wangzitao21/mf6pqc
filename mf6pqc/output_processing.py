"""Selected-output interpretation and durable result serialization."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from mf6pqc.properties import (
    extract_output_information as extract_output_information,
)
from mf6pqc.properties import (
    update_diffc as update_diffc,
)
from mf6pqc.properties import (
    update_porosity as update_porosity,
)

_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _atomic_file(path: Path, *, text: bool = False):
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        options = {"encoding": "utf-8", "newline": "\n"} if text else {}
        with os.fdopen(descriptor, "w" if text else "wb", **options) as handle:
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_save(path: Path, values: Any) -> None:
    with _atomic_file(path) as handle:
        np.save(handle, np.asarray(values))


def _atomic_write_text(path: Path, text: str) -> None:
    with _atomic_file(path, text=True) as handle:
        handle.write(text)


def environment_metadata() -> dict[str, Any]:
    """Return reproducibility information without importing optional backends."""
    import platform
    from importlib.metadata import PackageNotFoundError, version

    from mf6pqc._version import __version__

    packages = {"mf6pqc": __version__}
    for name in ("numpy", "modflowapi", "phreeqcrm", "flopy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


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
    if any(not isinstance(h, str) or not h.strip() or "\n" in h or "\r" in h for h in headings):
        raise ValueError("Headings must be nonempty single-line strings")
    # Validate user metadata before changing any existing result file.
    json.dumps(metadata, allow_nan=False)
    values = np.asarray(results, dtype=float)
    if values.ndim != 3:
        raise ValueError(f"results must have shape (time, output, cell); got {values.shape}")
    if values.shape[0] == 0 or values.shape[2] == 0:
        raise ValueError("Results must contain at least one frame and one cell")
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
            or np.any(porosity_values <= 0.0)
            or np.any(porosity_values > 1.0)
            or np.any(conductivity_values <= 0.0)
        ):
            raise ValueError("Porosity/K results contain invalid values")
    if if_update_diffc:
        diffusion_values = np.asarray(results_diffc, dtype=float)
        expected_shape = (max(0, values.shape[0] - 1), values.shape[2])
        if expected_shape[0] == 0 and diffusion_values.size == 0:
            diffusion_values = np.empty(expected_shape, dtype=float)
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
                    f"Energy field {name!r} has shape {field.shape}; expected {expected_shape}"
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
    # A manifest is the completion marker for this group of files. Remove the
    # previous marker only after validation, so interrupted writes cannot look
    # like a completed generation. Each array is independently atomic.
    manifest_path = Path(f"{base}_manifest.json")
    manifest_path.unlink(missing_ok=True)
    _atomic_save(result_path, values)
    _logger.info(f"Results saved to: {result_path}")

    headings_path = Path(f"{base}_headings.txt")
    _atomic_write_text(headings_path, "".join(f"{heading}\n" for heading in headings))
    _logger.info(f"Headings saved to: {headings_path}")

    saved_files = [result_path.name, headings_path.name]
    arrays = {}
    if times is not None:
        arrays["times"] = times
    if if_update_porosity_k:
        arrays.update(porosity=porosity_values, K=conductivity_values)
    if if_update_diffc:
        arrays["diffc"] = diffusion_values
    arrays.update(thermal_values)
    for name, field in arrays.items():
        path = Path(f"{base}_{name}.npy")
        _atomic_save(path, field)
        saved_files.append(path.name)
    energy_files = {name: f"{base.name}_{name}.npy" for name in thermal_values}

    manifest = {
        "schema_version": 1,
        "case_name": case_name,
        "result_shape": list(values.shape),
        "headings": list(headings),
        "files": saved_files,
        "has_porosity_and_k": bool(if_update_porosity_k),
        "has_diffusion": bool(if_update_diffc),
        "time_units": "days",
        "diffusion_time_axis": "results_times[1:]" if if_update_diffc else None,
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
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
    )
