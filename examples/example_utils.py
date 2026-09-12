"""Shared paths, input validation, and result helpers for the examples."""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from pathlib import Path

sys.dont_write_bytecode = True
REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if os.environ.get("MF6PQC_USE_INSTALLED") != "1" and str(REPOSITORY_DIR) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIR))

import numpy as np

MODFLOW_VERSION = "mf6.8.0"


def library_path(version: str = MODFLOW_VERSION) -> str:
    override = os.environ.get("MF6PQC_LIBMF6")
    if override:
        return str(Path(override).expanduser().resolve())
    name = {"win32": "libmf6.dll", "darwin": "libmf6.dylib"}.get(sys.platform, "libmf6.so")
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    return str((directory / name).expanduser().resolve())


def executable_path(version: str = MODFLOW_VERSION) -> str:
    override = os.environ.get("MF6PQC_MF6_EXE")
    if override:
        return str(Path(override).expanduser().resolve())
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    name = "mf6.exe" if sys.platform == "win32" else "mf6"
    return str((directory / name).expanduser().resolve())


def runtime_path(case_file: str | Path, kind: str, *, override: str | None = None) -> Path:
    """Resolve a case directory, honoring an explicit override or MF6PQC_RUN_ROOT."""
    if kind not in {"output", "simulation"}:
        raise ValueError("kind must be output or simulation")
    if override:
        return Path(override).expanduser().resolve()
    case_dir = Path(case_file).resolve().parent
    run_root = os.environ.get("MF6PQC_RUN_ROOT")
    base = Path(run_root).expanduser().resolve() / case_dir.name if run_root else case_dir
    return base / kind


def configure_logging() -> None:
    """Use the same concise progress-log format in each executable entry point."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def component_fields(
    species: Sequence[str], concentrations: np.ndarray, nxyz: int
) -> dict[str, np.ndarray]:
    """Split component-major concentrations into cell fields, checking size and names."""
    values = np.asarray(concentrations, dtype=float).reshape(-1)
    expected = len(species) * nxyz
    if values.size != expected:
        raise ValueError(f"Concentrations have {values.size} entries; expected {expected}")
    if len(set(species)) != len(species):
        raise ValueError("Component names must be unique")
    return dict(zip(species, values.reshape(len(species), nxyz), strict=True))


def boundary_values(species: Sequence[str], concentrations: np.ndarray) -> np.ndarray:
    """Normalize a boundary vector in the same component order as the initial fields."""
    values = np.asarray(concentrations, dtype=float).reshape(-1)
    if values.size != len(species):
        raise ValueError("Boundary concentrations must match the component count")
    return values


def read_headings(output_dir: str | Path) -> list[str]:
    path = Path(output_dir) / "results_headings.txt"
    headings = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    if not headings or any(not name for name in headings) or len(set(headings)) != len(headings):
        raise ValueError(f"Invalid component headings in {path}")
    return headings


def load_results(output_dir: str | Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    output_dir = Path(output_dir)
    values = np.load(output_dir / "results.npy", allow_pickle=False)
    headings = read_headings(output_dir)
    times = np.load(output_dir / "results_times.npy", allow_pickle=False)
    if values.ndim != 3 or times.ndim != 1 or values.shape[:2] != (len(times), len(headings)):
        raise ValueError(f"Inconsistent result dimensions in {output_dir}")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Result times must be finite and strictly increasing in {output_dir}")
    return (values, headings, times)


def time_indices(times: np.ndarray, targets: Sequence[float], *, atol: float = 1e-07) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    indices = []
    for target in targets:
        matches = np.flatnonzero(np.isclose(times, target, rtol=0.0, atol=atol))
        if len(matches) != 1:
            raise ValueError(f"Expected one saved state at {target}, found {len(matches)}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=int)


def restore_archives(directory: str | Path) -> None:
    directory = Path(directory).resolve()
    for archive_path in sorted(directory.glob("saved_results*.zip")):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (directory / member.filename).resolve()
                if not target.is_relative_to(directory):
                    raise ValueError("Archive member escapes the output directory")
                if not target.exists():
                    archive.extract(member, directory)


def file_digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_save(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.save(stream, values, allow_pickle=False)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def require_output_files(output_dir: str | Path, filenames: Sequence[str]) -> list[Path]:
    """Return required result paths, failing with all missing files in one message.

    Numerical validation belongs to MF6PQC.save_results(); this check confirms
    that the case-specific files expected by downstream analysis were written.
    """
    paths = [Path(output_dir) / filename for filename in filenames]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("MF6PQC reactive outputs were not saved: " + ", ".join(missing))
    return paths
