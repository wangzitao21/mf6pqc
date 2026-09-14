import logging
import warnings

import numpy as np

from mf6pqc.backends import (
    BackendFactory,
    NativeBackendFactory,
    initialize_phreeqcrm,
)
from mf6pqc.config import LEGACY_FIELDS, SimulationConfig
from mf6pqc.constants import SECONDS_PER_DAY, VM_MINERALS
from mf6pqc.coupling import (
    CouplingMethod,
    get_coupling_runner,
    run_sia,
    run_standard,
    run_strang,
    run_thermal_snia,
)
from mf6pqc.coupling.state import CouplingHooks
from mf6pqc.exceptions import ConfigurationError, CouplingError
from mf6pqc.input_processing import (
    setup_mixed_ic,
    setup_single_ic,
)
from mf6pqc.output_processing import (
    environment_metadata,
    save_results,
)
from mf6pqc.permeability import (
    BasePermeabilityUpdater,
)
from mf6pqc.permeability import (
    DensityCoupledKozenyCarmanUpdater as DensityCoupledKozenyCarmanUpdater,
)
from mf6pqc.permeability import PowerLawUpdater as PowerLawUpdater
from mf6pqc.properties import extract_output_information, get_calculated_density
from mf6pqc.results import ResultHistory
from mf6pqc.runtime import CellState, ChemistryState, RunStatus, TransportState, bind_aliases
from mf6pqc.types import ArrayLike, SIARateEvaluator
from mf6pqc.utils import require_integer

_logger = logging.getLogger(__name__)

VM_minerals = VM_MINERALS


class mf6pqc:
    """
    Reactive transport coupling MODFLOW 6 and PhreeqcRM.
    """

    def __init__(
        self,
        case_name: str = "temp_case",
        nxyz: int = 80,
        nthreads: int = 3,
        temperature: ArrayLike = 25.0,
        pressure: ArrayLike = 2.0,
        porosity: ArrayLike = 0.35,
        saturation: ArrayLike = 1.0,
        density: ArrayLike = 1.0,
        viscosity: ArrayLike = 1.0,
        d0: ArrayLike = 1.0e-9 * SECONDS_PER_DAY,
        print_chemistry_mask: ArrayLike = 0,
        componentH2O: bool = False,
        solution_density_volume: bool = False,
        db_path: str = None,
        pqi_path: str = None,
        modflow_dll_path: str = None,
        output_dir: str = None,
        workspace: str = None,
        if_update_porosity_K: bool = False,
        if_update_density: bool = False,
        if_update_diffc: bool = False,
        save_interval: int = 1,
        save_interval_offset: int = 0,
        save_steps: list[int] | None = None,
        reaction_steps: list[int] | None = None,
        progress_interval: int = 1000,
        fail_on_nonconvergence: bool = False,
        boundary_conductance_updates: dict | None = None,
        water_only_sink_rates: ArrayLike | None = None,
        use_phreeqc_calculated_density: bool = False,
        porosity_update_mask: ArrayLike = 1,
        sia_max_iterations: int = 2000,
        sia_rtol: float = 1.0e-4,
        sia_atol: float = 1.0e-9,
        sia_source_relaxation: float = 0.5,
        sia_density_relaxation: float = 0.5,
        sia_fail_on_nonconvergence: bool = False,
        sia_rate_evaluator: SIARateEvaluator | None = None,
        permeability_updater: BasePermeabilityUpdater | None = None,
        k33_ratio: float = 0.6,
        density_output_heading: str = "RHO",
        mineral_molar_volumes: dict[str, float] | None = None,
        backend_factory: BackendFactory | None = None,
        energy_enabled: bool = False,
        vsc_enabled: bool = False,
        flow_model_name: str = "gwf_model",
        energy_model_name: str = "gwe_model",
        npf_package_name: str = "NPF",
        vsc_package_name: str = "VSC",
        est_package_name: str = "EST",
        sync_gwe_temperature_to_phreeqc: bool = True,
        validate_initial_gwe_fields: bool = True,
        initial_gwe_field_tolerance: float = 1.0e-8,
        signed_components: tuple[str, ...] | list[str] = ("Charge",),
        *,
        result_storage: str = "memory",
        fail_on_porosity_clipping: bool = False,
    ):
        """
        Initialize a coupled MODFLOW 6 and PhreeqcRM simulator.
        Parameters
        ----------
        See class signature for configuration options.
        """
        parameters = {name: value for name, value in locals().items() if name != "self"}
        self._initialize(SimulationConfig.from_legacy(parameters))

    @classmethod
    def from_config(cls, config: SimulationConfig):
        """Construct a simulator from grouped, scientist-facing settings."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig instance")
        instance = cls.__new__(cls)
        instance._initialize(config)
        return instance

    def _initialize(self, config):
        self.config = config.validated()
        self.cells = CellState.from_config(self.config)
        self.chemistry = ChemistryState()
        self.transport = TransportState()
        self.lifecycle = RunStatus()
        self.history = ResultHistory()
        self.backend_factory = self.backend_factory or NativeBackendFactory()
        initialize_phreeqcrm(self)
        self.k_update_density_prev = self.density.copy()
        self.k_update_viscosity_prev = self.viscosity.copy()

    def setup(
        self,
        ic_map: dict,
        ic_map2: dict | None = None,
        fractions: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        Initialize chemical conditions and compute initial equilibrium.
        Parameters
        ----------
        ic_map : dict
            Mapping of module name to initial condition values.
        ic_map2 : dict | None
            Optional mapping for mixed initial conditions.
        fractions : ArrayLike | None
            Cell-wise fraction of ic_map (the first end member).
        Returns
        -------
        np.ndarray
            Initial concentrations after equilibrium.
        """
        self._ensure_open()
        if self.is_setup:
            warnings.warn(
                "setup() has already completed; returning the cached initial concentrations",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.initial_concentrations.copy()
        try:
            if ic_map2 is not None and fractions is not None:
                setup_mixed_ic(self.phreeqc_rm, self.nxyz, ic_map, ic_map2, fractions)
            elif ic_map2 is None and fractions is None:
                setup_single_ic(self.phreeqc_rm, self.nxyz, ic_map)
            else:
                raise ConfigurationError(
                    "ic_map2 and fractions must be provided together for mixed mode"
                )
            _logger.info("--- Running initial chemical equilibrium calculation ---")
            self.phreeqc_rm.SetTime(0.0 * SECONDS_PER_DAY)
            self.phreeqc_rm.SetTimeStep(0.0 * SECONDS_PER_DAY)
            self.phreeqc_rm.RunCells()
            initial = np.asarray(self.phreeqc_rm.GetConcentrations(), dtype=float).ravel()
            expected = self.nxyz * self.ncomps
            if initial.size != expected or not np.all(np.isfinite(initial)):
                raise CouplingError(
                    "Invalid initial concentration vector from PhreeqcRM: "
                    f"size={initial.size}, expected={expected}"
                )
            self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())
            if not self.headings:
                raise CouplingError(
                    "PhreeqcRM selected output has no headings; define SELECTED_OUTPUT/USER_PUNCH"
                )
            raw_selected = np.asarray(self.phreeqc_rm.GetSelectedOutput(), dtype=float)
            if raw_selected.size != len(self.headings) * self.nxyz:
                raise CouplingError(
                    "Selected output size does not match headings and nxyz: "
                    f"{raw_selected.size} != {len(self.headings)} * {self.nxyz}"
                )
            self.selected_output = raw_selected.reshape(-1, self.nxyz)
            if not np.all(np.isfinite(self.selected_output)):
                raise CouplingError("Initial selected output contains non-finite values")
            if self.if_update_density:
                if not self.use_phreeqc_calculated_density:
                    matches = [
                        index
                        for index, heading in enumerate(self.headings)
                        if heading.casefold() == self.density_output_heading.casefold()
                    ]
                    if len(matches) != 1:
                        raise ConfigurationError(
                            f"Density feedback requires exactly one {self.density_output_heading!r} selected-output heading"
                        )
                    self.chemistry.density_row = matches[0]
                get_calculated_density(self)
            if self.if_update_porosity_K:
                self.output_indices, self.mineral_volumes, self.d_mineral_names = (
                    extract_output_information(self.headings, self.mineral_molar_volumes)
                )
                if self.output_indices.size == 0:
                    raise ConfigurationError(
                        "Porosity feedback is enabled, but selected output contains no "
                        "d_<mineral> headings"
                    )
            self.initial_concentrations = initial.copy()
            self.results.append(self.selected_output.copy())
            self.result_times.append(0.0)
            self.is_setup = True
            return initial
        except BaseException:
            self.finalize()
            raise

    def run(
        self, method: CouplingMethod | str | None = None, *, hooks: CouplingHooks | None = None
    ) -> None:
        """
        Advance a configured SNIA, SIA, Strang, or ThermalSNIA simulation.
        Parameters
        ----------
        None
            Uses instance configuration and state.
        Returns
        -------
        None
            Advances the simulation and stores results.
        """
        if method is None:
            self._run_coupling(run_standard, CouplingMethod.SNIA, hooks=hooks)
            return
        normalized, runner = get_coupling_runner(method)
        self._run_coupling(runner, normalized, hooks=hooks)

    def run_SNIA(self, *, hooks: CouplingHooks | None = None) -> None:
        """Run the sequential non-iterative coupling loop explicitly."""
        self._run_coupling(run_standard, CouplingMethod.SNIA, hooks=hooks)

    def run_SIA(self, *, hooks: CouplingHooks | None = None) -> None:
        """
        Run the SIA coupling loop with source feedback.
        Parameters
        ----------
        None
            Uses instance configuration and state.
        Returns
        -------
        None
            Advances the simulation and stores results.
        """
        self._run_coupling(run_sia, CouplingMethod.SIA, hooks=hooks)

    def run_Strang(self, *, hooks: CouplingHooks | None = None) -> None:
        """Run symmetric transport-reaction-transport Strang splitting."""
        self._run_coupling(run_strang, CouplingMethod.STRANG, hooks=hooks)

    def run_ThermalSNIA(self, *, hooks: CouplingHooks | None = None) -> None:
        """Run explicit GWF-GWT-GWE-VSC reactive transport."""
        self._run_coupling(run_thermal_snia, CouplingMethod.THERMAL_SNIA, hooks=hooks)

    def _run_coupling(
        self,
        runner,
        method: CouplingMethod | str | None = None,
        *,
        hooks: CouplingHooks | None = None,
    ) -> None:
        """Apply lifecycle guards around a coupling algorithm."""
        self._ensure_open()
        if self._run_active:
            raise CouplingError("A coupling run is already active")
        if self._run_completed:
            raise CouplingError(
                "This simulator has already completed a run; create a new instance "
                "for another simulation"
            )
        if self.reaction_steps is not None and method is not CouplingMethod.SNIA:
            raise ConfigurationError(
                "reaction_steps is currently implemented only for SNIA; "
                f"received coupling method {method!r}"
            )
        is_thermal = method is CouplingMethod.THERMAL_SNIA
        if self.energy_enabled and not is_thermal:
            raise ConfigurationError(
                "energy_enabled=True requires method='ThermalSNIA'; legacy "
                "SNIA/SIA/Strang paths intentionally remain unchanged"
            )
        if is_thermal and not self.energy_enabled:
            raise ConfigurationError("ThermalSNIA requires energy_enabled=True")
        if hooks is not None:
            if not isinstance(hooks, CouplingHooks):
                raise ConfigurationError("hooks must be a CouplingHooks instance")
            if method is not CouplingMethod.SNIA and (
                hooks.on_transport is not None or hooks.on_reaction is not None
            ):
                raise ConfigurationError("Transport and reaction hooks require SNIA")
        self._coupling_hooks = hooks
        self._run_active = True
        if isinstance(method, CouplingMethod):
            self.last_coupling_method = method.value
        elif method is not None:
            self.last_coupling_method = str(method)
        else:
            self.last_coupling_method = runner.__name__
        try:
            runner(self)
        except BaseException:
            self.finalize()
            raise
        else:
            self._run_completed = True
        finally:
            self._run_active = False
            self._coupling_hooks = None

    def save_results(self, filename: str = None) -> None:
        """
        Save selected outputs and transport properties to disk.
        Parameters
        ----------
        filename : str | None
            Optional base filename for results.
        Returns
        -------
        None
            Writes results to output directory.
        """
        from mf6pqc.energy import energy_result_payload

        save_results(
            self.output_dir,
            self.case_name,
            self.headings,
            self.results,
            self.results_porosity,
            self.results_K,
            self.results_diffc,
            self.if_update_porosity_K,
            self.if_update_diffc,
            filename,
            result_times=self.result_times,
            metadata={
                "completed": self._run_completed,
                "nxyz": self.nxyz,
                "nthreads": self.nthreads,
                "chemistry_backend": type(self.backend_factory).__name__,
                "chemistry_processes": min(
                    self.nxyz, getattr(self.backend_factory, "processes", 1)
                ),
                "components": self.components,
                "environment": environment_metadata(),
                "inputs": self.input_provenance,
                "coupling_method": self.last_coupling_method,
                "logical_steps": self.final_time_step_index,
                "wall_time_seconds": self.last_run_wall_time_seconds,
                "modflow_convergence_failures": self.modflow_convergence_failures,
                "sia_iterations": self.sia_iterations,
                "sia_convergence_failures": self.sia_convergence_failures,
                "sia_diagnostics": self.sia_diagnostics,
                "porosity_clipping": self.porosity_clipping,
                "converged": self._run_completed
                and not (self.modflow_convergence_failures or self.sia_convergence_failures),
                "result_storage": self.result_storage,
            },
            energy_results=energy_result_payload(self),
        )

    def finalize(self) -> None:
        """
        Finalize simulation and release resources.
        Parameters
        ----------
        None
            Uses instance configuration and state.
        Returns
        -------
        None
            Closes MODFLOW 6 and PhreeqcRM resources.
        """
        if self._modflow_finalized and self._chemistry_finalized:
            return
        _logger.info("--- Finalizing simulation, releasing resources ---")
        if self.modflow_api is None:
            self._modflow_finalized = True
        elif not self._modflow_finalized:
            try:
                self.modflow_api.finalize()
                _logger.info("MODFLOW API closed")
            except Exception as exc:
                warnings.warn(
                    f"MODFLOW API finalization failed: {exc}",
                    ResourceWarning,
                    stacklevel=2,
                )
            finally:
                self._modflow_finalized = True
        if self.phreeqc_rm is None:
            self._chemistry_finalized = True
        elif not self._chemistry_finalized:
            for operation in ("CloseFiles", "MpiWorkerBreak"):
                try:
                    getattr(self.phreeqc_rm, operation)()
                except Exception as exc:
                    warnings.warn(
                        f"PhreeqcRM {operation} failed: {exc}",
                        ResourceWarning,
                        stacklevel=2,
                    )
            self._chemistry_finalized = True
        self.phreeqc_rm = None
        self.modflow_api = None
        self.sim = None
        self.is_setup = False

    def _ensure_open(self) -> None:
        """Reject reuse of closed native solver state."""
        if getattr(self, "_chemistry_finalized", False) or getattr(
            self, "_modflow_finalized", False
        ):
            raise CouplingError("This simulator is finalized; create a new instance")

    def __enter__(self):
        """Return this simulator for use as a context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """Release native resources when leaving a context block."""
        self.finalize()
        return False

    def get_components(self) -> list:
        """
        Retrieve reactive component names.
        Parameters
        ----------
        None
            Uses the internal PhreeqcRM object.
        Returns
        -------
        list
            List of component names.
        """
        return list(self.components)

    def get_initial_concentrations(self, number: float) -> np.ndarray:
        """
        Retrieve boundary concentration for a single value.
        Parameters
        ----------
        number : float
            Value used to create a boundary concentration.
        Returns
        -------
        np.ndarray
            Boundary concentration vector.
        """
        self._ensure_open()
        number = require_integer("solution number", number, minimum=0)
        bc1 = np.full(1, number, dtype=np.int32)
        return self.phreeqc_rm.InitialPhreeqc2Concentrations(bc1)


# PEP 8 public name for new code; the historical lowercase class remains the
# implementation so existing imports, notebooks, and serialized metadata keep
# working unchanged.
MF6PQC = mf6pqc

bind_aliases(
    mf6pqc,
    {
        **{name: f"config.{path}" for name, path in LEGACY_FIELDS.items()},
        **{name: f"cells.{name}" for name in CellState.__dataclass_fields__},
        **{
            name: f"chemistry.{name}"
            for name in ChemistryState.__dataclass_fields__
            if name != "backend"
        },
        **{
            name: f"transport.{name}"
            for name in TransportState.__dataclass_fields__
            if name not in {"backend", "simulation"}
        },
        **{name: f"lifecycle.{name}" for name in RunStatus.__dataclass_fields__},
        **{name: f"history.{name}" for name in ResultHistory.__dataclass_fields__},
        "phreeqc_rm": "chemistry.backend",
        "modflow_api": "transport.backend",
        "sim": "transport.simulation",
        "perm_updater": "config.feedback.permeability_updater",
        "_run_active": "lifecycle.active",
        "_run_completed": "lifecycle.completed",
        "_chemistry_finalized": "lifecycle.chemistry_finalized",
        "_modflow_finalized": "lifecycle.modflow_finalized",
        "_coupling_hooks": "lifecycle.hooks",
    },
)
