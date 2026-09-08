"""Shared path conventions for the source-tree examples, outside the runtime package."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPOSITORY_DIR = Path(__file__).resolve().parent.parent
if os.environ.get("MF6PQC_USE_INSTALLED") != "1":
    sys.path.insert(0, str(REPOSITORY_DIR))


def library_path(version: str = "mf6.7.0") -> str:
    """Use an explicit library, an installed binary directory, or the local benchmark version."""
    override = os.environ.get("MF6PQC_LIBMF6")
    if override:
        return str(Path(override).expanduser().resolve())
    name = (
        "libmf6.dll"
        if sys.platform == "win32"
        else "libmf6.dylib"
        if sys.platform == "darwin"
        else "libmf6.so"
    )
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    return str((directory / name).expanduser().resolve())


def executable_path(version: str = "mf6.7.0") -> str:
    """Resolve the optional standalone MODFLOW executable used by FloPy metadata."""
    override = os.environ.get("MF6PQC_MF6_EXE")
    if override:
        return str(Path(override).expanduser().resolve())
    directory = Path(os.environ.get("MF6PQC_BIN", REPOSITORY_DIR / "bin" / version))
    return str(
        (directory / ("mf6.exe" if sys.platform == "win32" else "mf6")).expanduser().resolve()
    )


def runtime_path(case_file: str, kind: str) -> Path:
    """Locate generated files; an optional run root permits isolated regression runs."""
    if kind not in {"output", "simulation"}:
        raise ValueError("kind must be output or simulation")
    case = Path(case_file).resolve().parent
    override = os.environ.get("MF6PQC_RUN_ROOT")
    base = Path(override).expanduser().resolve() / case.name if override else case
    return base / kind


def configure_logging() -> None:
    """Show solver progress only when an example is explicitly executed."""
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
