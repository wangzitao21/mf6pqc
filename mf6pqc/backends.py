"""Backend creation and lifecycle boundaries.

Only this module knows how the concrete ``phreeqcrm`` and ``modflowapi``
packages are constructed.  Coupling algorithms operate on their public API
surface, which keeps the numerical loop testable without loading native
libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mf6pqc.constants import (
    PHREEQCRM_REBALANCE_FRACTION,
    PHREEQCRM_TIME_CONVERSION,
    PHREEQCRM_UNITS,
)
from mf6pqc.exceptions import BackendError, ConfigurationError


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
    print("--- Initializing PhreeqcRM ---")
    database = _require_file(sim.db_path, "db_path")
    chemistry_input = _require_file(sim.pqi_path, "pqi_path")
    if sim.output_dir is None:
        raise ConfigurationError("output_dir must be provided")
    output_dir = str(Path(sim.output_dir).expanduser().resolve())
    os.makedirs(output_dir, exist_ok=True)
    sim.output_dir = output_dir

    chemistry = None
    files_open = False
    try:
        chemistry = sim.backend_factory.create_phreeqcrm(sim.nxyz, sim.nthreads)
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
        chemistry.SetComponentH2O(sim.componentH2O)
        chemistry.UseSolutionDensityVolume(sim.solution_density_volume)
        chemistry.SetRebalanceFraction(PHREEQCRM_REBALANCE_FRACTION)
        print(f"Loading Phreeqc database: {database}")
        chemistry.LoadDatabase(database)
        chemistry.SetPrintChemistryOn(True, False, False)
        print(f"Running chemistry definition file: {chemistry_input}")
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
        print(f"List of reactive chemical components: {sim.components}")
        chemistry.SetScreenOn(False)
        chemistry.SetSelectedOutputOn(True)
    except Exception as exc:
        if chemistry is not None:
            if files_open:
                try:
                    chemistry.CloseFiles()
                except Exception:
                    pass
            try:
                chemistry.MpiWorkerBreak()
            except Exception:
                pass
        sim.phreeqc_rm = None
        if isinstance(exc, (ConfigurationError, BackendError)):
            raise
        raise BackendError(f"Failed to initialize PhreeqcRM: {exc}") from exc


def initialize_modflow6(sim) -> None:
    """Create and initialize the MODFLOW 6 backend for ``sim``."""
    print("--- Initializing MODFLOW 6 ---")
    dll_path = _require_file(sim.modflow_dll_path, "modflow_dll_path")
    workspace = _require_directory(sim.workspace, "workspace")
    sim.workspace = workspace
    print(f"Working directory: {workspace}")
    api = None
    initialized = False
    try:
        api = sim.backend_factory.create_modflow_api(dll_path, workspace)
        api.initialize()
        initialized = True
        sim.modflow_api = api
        sim.sim = sim.backend_factory.load_modflow_simulation(api)
    except Exception as exc:
        if api is not None and initialized:
            try:
                api.finalize()
            except Exception:
                pass
        sim.modflow_api = None
        raise BackendError(
            "Failed to initialize MODFLOW 6. "
            f"DLL: {dll_path}; workspace: {workspace}; reason: {exc}"
        ) from exc
