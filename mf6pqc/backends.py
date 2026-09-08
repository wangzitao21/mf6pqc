"""Backend creation and lifecycle boundaries.

Only this module knows how the concrete ``phreeqcrm`` and ``modflowapi``
packages are constructed.  Coupling algorithms operate on their public API
surface, which keeps the numerical loop testable without loading native
libraries.
"""

from __future__ import annotations

import contextlib
import logging
import numbers
import os
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mf6pqc.constants import (
    PHREEQCRM_REBALANCE_FRACTION,
    PHREEQCRM_TIME_CONVERSION,
    PHREEQCRM_UNITS,
)
from mf6pqc.exceptions import BackendError, ConfigurationError

_logger = logging.getLogger(__name__)


class CheckedPhreeqcRM:
    """Turn negative PhreeqcRM status codes into Python exceptions.

    PhreeqcRM's default error mode returns a status instead of raising. Getter
    values (which can legitimately be negative) retain their native semantics.
    Methods are cached so the transport loop adds no repeated wrapper creation.
    The underlying solver remains accessible through ``backend``.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def __getattr__(self, name: str):
        attribute = getattr(self.backend, name)
        if not callable(attribute) or name.startswith("Get"):
            return attribute

        backend = self.backend

        @wraps(attribute)
        def checked(*args, **kwargs):
            result = attribute(*args, **kwargs)
            if isinstance(result, numbers.Integral) and result < 0:
                detail = ""
                with contextlib.suppress(AttributeError, RuntimeError):
                    detail = str(backend.GetErrorString()).strip()
                raise BackendError(f"PhreeqcRM {name} failed (status {result}): {detail}")
            return result

        setattr(self, name, checked)
        return checked


@runtime_checkable
class BackendFactory(Protocol):
    """Construction seam for the native scientific backends."""

    def create_phreeqcrm(self, nxyz: int, nthreads: int) -> Any:
        """Create an unconfigured PhreeqcRM instance."""

    def create_modflow_api(self, dll_path: str, workspace: str) -> Any:
        """Create an uninitialized MODFLOW 6 API instance."""

    def load_modflow_simulation(self, modflow_api: Any) -> Any:
        """Load the high-level API simulation wrapper."""


@dataclass(frozen=True, slots=True)
class NativeBackendFactory:
    """Default factory backed by the installed native Python packages."""

    def create_phreeqcrm(self, nxyz: int, nthreads: int) -> Any:
        import phreeqcrm

        return phreeqcrm.PhreeqcRM(nxyz, nthreads)

    def create_modflow_api(self, dll_path: str, workspace: str) -> Any:
        import modflowapi

        return modflowapi.ModflowApi(dll_path, working_directory=workspace)

    def load_modflow_simulation(self, modflow_api: Any) -> Any:
        import modflowapi

        return modflowapi.extensions.ApiSimulation.load(modflow_api)


def file_fingerprint(path: str | Path) -> dict[str, str | int]:
    """Identify the exact solver input without embedding its contents in output."""
    import hashlib

    resolved = Path(path).resolve()
    with resolved.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(resolved), "sha256": digest, "size_bytes": resolved.stat().st_size}


def _require_file(path: str | os.PathLike[str] | None, label: str) -> str:
    if path is None:
        raise ConfigurationError(f"{label} must be provided")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigurationError(f"{label} does not exist or is not a file: {resolved}")
    return str(resolved)


def _require_directory(path: str | os.PathLike[str] | None, label: str) -> str:
    if path is None:
        raise ConfigurationError(f"{label} must be provided")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ConfigurationError(f"{label} does not exist or is not a directory: {resolved}")
    return str(resolved)


def initialize_phreeqcrm(sim) -> None:
    """Create and fully configure the chemistry backend for ``sim``."""
    _logger.info("--- Initializing PhreeqcRM ---")
    database = _require_file(sim.db_path, "db_path")
    chemistry_input = _require_file(sim.pqi_path, "pqi_path")
    if sim.output_dir is None:
        raise ConfigurationError("output_dir must be provided")
    output_dir = str(Path(sim.output_dir).expanduser().resolve())
    os.makedirs(output_dir, exist_ok=True)
    sim.output_dir = output_dir

    sim.input_provenance = {
        "database": file_fingerprint(database),
        "chemistry_input": file_fingerprint(chemistry_input),
    }
    chemistry = None
    files_open = False
    try:
        chemistry = CheckedPhreeqcRM(sim.backend_factory.create_phreeqcrm(sim.nxyz, sim.nthreads))
        sim.phreeqc_rm = chemistry
        prefix = os.path.join(output_dir, f"{sim.case_name}_prm")
        chemistry.SetFilePrefix(prefix)
        chemistry.OpenFiles()
        files_open = True
        chemistry.SetUnitsSolution(PHREEQCRM_UNITS["solution"])
        chemistry.SetUnitsPPassemblage(PHREEQCRM_UNITS["ppassemblage"])
        chemistry.SetUnitsExchange(PHREEQCRM_UNITS["exchange"])
        chemistry.SetUnitsSurface(PHREEQCRM_UNITS["surface"])
        chemistry.SetUnitsGasPhase(PHREEQCRM_UNITS["gas_phase"])
        chemistry.SetUnitsSSassemblage(PHREEQCRM_UNITS["ssassemblage"])
        chemistry.SetUnitsKinetics(PHREEQCRM_UNITS["kinetics"])
        chemistry.SetTimeConversion(PHREEQCRM_TIME_CONVERSION)
        chemistry.SetTemperature(sim.temperature)
        chemistry.SetPressure(sim.pressure)
        chemistry.SetPorosity(sim.porosity)
        chemistry.SetSaturation(sim.saturation)
        chemistry.SetDensityUser(sim.density)
        chemistry.SetPrintChemistryMask(sim.print_chemistry_mask.astype("int32"))
        chemistry.SetComponentH2O(sim.componentH2O)
        chemistry.UseSolutionDensityVolume(sim.solution_density_volume)
        chemistry.SetRebalanceFraction(PHREEQCRM_REBALANCE_FRACTION)
        _logger.info(f"Loading Phreeqc database: {database}")
        chemistry.LoadDatabase(database)
        chemistry.SetPrintChemistryOn(True, False, False)
        _logger.info(f"Running chemistry definition file: {chemistry_input}")
        chemistry.RunFile(True, True, True, chemistry_input)
        chemistry.RunString(True, False, True, "DELETE; -all")
        sim.ncomps = chemistry.FindComponents()
        sim.components = list(chemistry.GetComponents())
        if sim.ncomps != len(sim.components):
            raise BackendError(
                "PhreeqcRM component count does not match GetComponents(): "
                f"{sim.ncomps} != {len(sim.components)}"
            )
        if not sim.components:
            raise BackendError("PhreeqcRM did not report any transport components")
        _logger.info(f"List of reactive chemical components: {sim.components}")
        chemistry.SetScreenOn(False)
        chemistry.SetSelectedOutputOn(True)
    except BaseException as exc:
        if chemistry is not None:
            if files_open:
                with contextlib.suppress(Exception):
                    chemistry.CloseFiles()
            with contextlib.suppress(Exception):
                chemistry.MpiWorkerBreak()
        sim.phreeqc_rm = None
        if not isinstance(exc, Exception) or isinstance(exc, (ConfigurationError, BackendError)):
            raise
        raise BackendError(f"Failed to initialize PhreeqcRM: {exc}") from exc


def validate_modflow_workspace(workspace: str | Path) -> None:
    """Reject unsupported time units and adaptive stepping before native initialization."""
    import shlex

    def records(path):
        for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
            lexer = shlex.shlex(line, posix=True)
            lexer.whitespace_split = True
            lexer.escape = ""  # Preserve Windows paths inside MODFLOW input.
            tokens = list(lexer)
            if tokens:
                yield tokens

    workspace = Path(workspace)
    namefile = workspace / "mfsim.nam"
    if not namefile.is_file():
        raise ConfigurationError(f"MODFLOW simulation name file is missing: {namefile}")
    tdis_files = [row[1] for row in records(namefile) if row[0].upper() == "TDIS6" and len(row) > 1]
    if len(tdis_files) != 1:
        raise ConfigurationError("mfsim.nam must define exactly one TDIS6 file")
    tdis = workspace / tdis_files[0]
    if not tdis.is_file():
        raise ConfigurationError(f"TDIS file is missing: {tdis}")
    units = None
    for row in records(tdis):
        if row[0].upper() == "ATS6":
            raise ConfigurationError("ATS is not supported; use a static TDIS schedule")
        if row[0].upper() == "TIME_UNITS" and len(row) > 1:
            units = row[1].upper()
    if units != "DAYS":
        raise ConfigurationError(
            "MF6PQC currently requires TDIS TIME_UNITS DAYS; chemistry time is converted to seconds"
        )


def initialize_modflow6(sim) -> None:
    """Create and initialize the MODFLOW 6 backend for ``sim``."""
    _logger.info("--- Initializing MODFLOW 6 ---")
    dll_path = _require_file(sim.modflow_dll_path, "modflow_dll_path")
    workspace = _require_directory(sim.workspace, "workspace")
    validate_modflow_workspace(workspace)
    sim.input_provenance["modflow_library"] = file_fingerprint(dll_path)
    sim.workspace = workspace
    _logger.info(f"Working directory: {workspace}")
    api = None
    initialized = False
    try:
        api = sim.backend_factory.create_modflow_api(dll_path, workspace)
        api.initialize()
        initialized = True
        sim.modflow_api = api
        sim.sim = sim.backend_factory.load_modflow_simulation(api)
    except BaseException as exc:
        if api is not None and initialized:
            with contextlib.suppress(Exception):
                api.finalize()
        sim.modflow_api = None
        if not isinstance(exc, Exception):
            raise
        raise BackendError(
            "Failed to initialize MODFLOW 6. "
            f"DLL: {dll_path}; workspace: {workspace}; reason: {exc}"
        ) from exc
