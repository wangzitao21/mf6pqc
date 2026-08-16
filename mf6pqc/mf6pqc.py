import numpy as np
import warnings

from mf6pqc.backends import (
    BackendFactory,
    NativeBackendFactory,
    initialize_modflow6,
    initialize_phreeqcrm,
)
from mf6pqc.types import ArrayLike, SIARateEvaluator
from mf6pqc.constants import VM_MINERALS, SECONDS_PER_DAY
from mf6pqc.utils import ensure_array, get_species_slice
from mf6pqc.input_processing import (
    create_ic_array_from_map,
    setup_single_ic,
    setup_mixed_ic,
)
from mf6pqc.output_processing import (
    extract_output_information,
    update_porosity,
    update_diffc,
    save_results,
)
from mf6pqc.coupling import (
    CouplingMethod,
    get_coupling_runner,
    run_standard,
    run_sia,
    run_strang,
    run_thermal_snia,
)
from mf6pqc.permeability import (
    BasePermeabilityUpdater,
    FluidAdjustedKozenyCarmanUpdater,
    KozenyCarmanUpdater,
    DensityCoupledKozenyCarmanUpdater,
    PowerLawUpdater,
)
from mf6pqc.exceptions import ConfigurationError, CouplingError
from mf6pqc.config import SimulationConfig

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
        print_chemistry_mask: ArrayLike = 1,
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
    ):
        """
        Initialize a coupled MODFLOW 6 and PhreeqcRM simulator.
        Parameters
        ----------
        See class signature for configuration options.
        """
        self.backend_factory = backend_factory or NativeBackendFactory()
        self._set_core_config(
            case_name,
            nxyz,
            nthreads,
            componentH2O,
            solution_density_volume,
            db_path,
            pqi_path,
            modflow_dll_path,
            output_dir,
            workspace,
            save_interval,
            save_interval_offset,
            save_steps,
            reaction_steps,
            progress_interval,
        )
        self._set_runtime_flags(if_update_porosity_K, if_update_density, if_update_diffc)
        self._set_component_domains(signed_components)
        self._set_energy_config(
            energy_enabled,
            vsc_enabled,
            flow_model_name,
            energy_model_name,
            npf_package_name,
            vsc_package_name,
            est_package_name,
            sync_gwe_temperature_to_phreeqc,
            validate_initial_gwe_fields,
            initial_gwe_field_tolerance,
        )
        self.fail_on_nonconvergence = fail_on_nonconvergence
        self.use_phreeqc_calculated_density = use_phreeqc_calculated_density
        self.density_output_heading = str(density_output_heading)
        self.mineral_molar_volumes = dict(VM_MINERALS)
        if mineral_molar_volumes:
            self.mineral_molar_volumes.update(mineral_molar_volumes)
        for mineral, molar_volume in self.mineral_molar_volumes.items():
            if (
                not isinstance(mineral, str)
                or not mineral
                or not np.isfinite(molar_volume)
                or molar_volume <= 0.0
            ):
                raise ConfigurationError(
                    "mineral_molar_volumes must map non-empty names to positive "
                    "finite values in L/mol"
                )
        if not np.isfinite(k33_ratio) or k33_ratio <= 0.0:
            raise ConfigurationError("k33_ratio must be finite and positive")
        self.k33_ratio = float(k33_ratio)
        self._set_sia_config(
            sia_max_iterations,
            sia_rtol,
            sia_atol,
            sia_source_relaxation,
            sia_density_relaxation,
            sia_fail_on_nonconvergence,
            sia_rate_evaluator,
        )
        # Optional fixed-head GHB faces whose conductance must track the
        # adjacent cell K.  Entries are keyed by MODFLOW package name and
        # contain ``cell_index``, ``distance`` and optional ``area``.
        self.boundary_conductance_updates = dict(boundary_conductance_updates or {})
        self._init_state_containers()
        self._init_fields(
            temperature,
            pressure,
            porosity,
            saturation,
            density,
            viscosity,
            d0,
            porosity_update_mask,
            print_chemistry_mask,
            water_only_sink_rates,
        )
        if permeability_updater is not None and not isinstance(
            permeability_updater, BasePermeabilityUpdater
        ):
            raise TypeError(
                "permeability_updater must implement BasePermeabilityUpdater"
            )
        if permeability_updater is not None:
            self.perm_updater = permeability_updater
        else:
            self.perm_updater = KozenyCarmanUpdater()
        self._validate_feature_combinations()
        self._initialize_phreeqcrm()
        self.k_update_density_prev = self.density.copy()
        self.k_update_viscosity_prev = self.viscosity.copy()

    def _set_component_domains(
        self, signed_components: tuple[str, ...] | list[str]
    ) -> None:
        """Store components whose valid concentration domain includes negatives."""
        if isinstance(signed_components, (str, bytes)):
            raise TypeError("signed_components must be a sequence of component names")
        try:
            names = tuple(signed_components)
        except TypeError as exc:
            raise TypeError(
                "signed_components must be a sequence of component names"
            ) from exc
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError(
                "signed_components must contain only non-empty component names"
            )
        self.signed_components = frozenset(name.strip().casefold() for name in names)

    @classmethod
    def from_config(cls, config: SimulationConfig):
        """Construct a simulator from grouped, scientist-facing settings."""
        if not isinstance(config, SimulationConfig):
            raise TypeError("config must be a SimulationConfig instance")
        return cls(**config.to_legacy_kwargs())

    def _set_sia_config(
        self,
        max_iterations: int,
        rtol: float,
        atol: float,
        source_relaxation: float,
        density_relaxation: float,
        fail_on_nonconvergence: bool,
        rate_evaluator: SIARateEvaluator | None,
    ) -> None:
        """Validate and store controls for the sequential iterative method."""
        if max_iterations <= 0:
            raise ValueError("sia_max_iterations must be a positive integer")
        if rtol < 0.0 or atol < 0.0:
            raise ValueError("SIA convergence tolerances must be nonnegative")
        if not 0.0 < source_relaxation <= 1.0:
            raise ValueError("sia_source_relaxation must be in (0, 1]")
        if not 0.0 < density_relaxation <= 1.0:
            raise ValueError("sia_density_relaxation must be in (0, 1]")
        if rate_evaluator is not None and not callable(rate_evaluator):
            raise TypeError("sia_rate_evaluator must be callable or None")
        self.sia_max_iterations = int(max_iterations)
        self.sia_rtol = float(rtol)
        self.sia_atol = float(atol)
        self.sia_source_relaxation = float(source_relaxation)
        self.sia_density_relaxation = float(density_relaxation)
        self.sia_fail_on_nonconvergence = bool(fail_on_nonconvergence)
        self.sia_rate_evaluator = rate_evaluator

    def _set_core_config(
        self,
        case_name: str,
        nxyz: int,
        nthreads: int,
        componentH2O: bool,
        solution_density_volume: bool,
        db_path: str,
        pqi_path: str,
        modflow_dll_path: str,
        output_dir: str,
        workspace: str,
        save_interval: int,
        save_interval_offset: int,
        save_steps: list[int] | None,
        reaction_steps: list[int] | None,
        progress_interval: int,
    ) -> None:
        """
        Set core configuration attributes.
        Parameters
        ----------
        See class signature for configuration options.
        Returns
        -------
        None
            Updates basic configuration fields.
        """
        if not isinstance(case_name, str) or not case_name.strip():
            raise ConfigurationError("case_name must be a non-empty string")
        if isinstance(nxyz, bool) or int(nxyz) != nxyz or int(nxyz) <= 0:
            raise ConfigurationError("nxyz must be a positive integer")
        if isinstance(nthreads, bool) or int(nthreads) != nthreads or int(nthreads) <= 0:
            raise ConfigurationError("nthreads must be a positive integer")
        if isinstance(save_interval, bool) or int(save_interval) != save_interval:
            raise ConfigurationError("save_interval must be an integer")
        if int(save_interval) <= 0:
            raise ConfigurationError("save_interval must be positive")
        if (
            isinstance(save_interval_offset, bool)
            or int(save_interval_offset) != save_interval_offset
        ):
            raise ConfigurationError("save_interval_offset must be an integer")
        self.case_name = case_name.strip()
        self.nxyz = int(nxyz)
        self.nthreads = int(nthreads)
        self.componentH2O = bool(componentH2O)
        self.solution_density_volume = bool(solution_density_volume)
        self.db_path = db_path
        self.pqi_path = pqi_path
        self.modflow_dll_path = modflow_dll_path
        self.output_dir = output_dir
        self.workspace = workspace
        self.save_interval = int(save_interval)
        self.save_interval_offset = int(save_interval_offset)
        if save_steps is None:
            self.save_steps = None
        else:
            normalised_steps = frozenset(int(step) for step in save_steps)
            if any(step <= 0 for step in normalised_steps):
                raise ValueError("save_steps must contain positive, one-based step numbers")
            self.save_steps = normalised_steps
        if reaction_steps is None:
            self.reaction_steps = None
        else:
            normalised_reaction_steps = frozenset(
                int(step) for step in reaction_steps
            )
            if not normalised_reaction_steps or any(
                step <= 0 for step in normalised_reaction_steps
            ):
                raise ValueError(
                    "reaction_steps must contain positive, one-based step numbers"
                )
            self.reaction_steps = normalised_reaction_steps
        if (
            isinstance(progress_interval, bool)
            or int(progress_interval) != progress_interval
            or int(progress_interval) <= 0
        ):
            raise ValueError("progress_interval must be a positive integer")
        self.progress_interval = int(progress_interval)

    def _set_runtime_flags(
        self, if_update_porosity_K: bool, if_update_density: bool, if_update_diffc: bool
    ) -> None:
        """
        Set feature flags controlling feedback updates.
        Parameters
        ----------
        if_update_porosity_K : bool
            Whether to update porosity and permeability.
        if_update_density : bool
            Whether to update density feedback.
        if_update_diffc : bool
            Whether to update diffusion coefficients.
        Returns
        -------
        None
            Stores runtime flags on the instance.
        """
        self.if_update_porosity_K = if_update_porosity_K
        self.if_update_density = if_update_density
        self.if_update_diffc = if_update_diffc

    def _set_energy_config(
        self,
        energy_enabled: bool,
        vsc_enabled: bool,
        flow_model_name: str,
        energy_model_name: str,
        npf_package_name: str,
        vsc_package_name: str,
        est_package_name: str,
        sync_temperature: bool,
        validate_initial_fields: bool,
        initial_field_tolerance: float,
    ) -> None:
        """Validate and store opt-in GWE/VSC coupling controls."""
        self.energy_enabled = bool(energy_enabled)
        self.vsc_enabled = bool(vsc_enabled)
        if self.vsc_enabled and not self.energy_enabled:
            raise ConfigurationError("vsc_enabled=True requires energy_enabled=True")
        names = {
            "flow_model_name": flow_model_name,
            "energy_model_name": energy_model_name,
            "npf_package_name": npf_package_name,
            "vsc_package_name": vsc_package_name,
            "est_package_name": est_package_name,
        }
        for label, value in names.items():
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"{label} must be a non-empty string")
            setattr(self, label, value.strip())
        if (
            not np.isfinite(initial_field_tolerance)
            or initial_field_tolerance < 0.0
        ):
            raise ConfigurationError(
                "initial_gwe_field_tolerance must be finite and nonnegative"
            )
        self.sync_gwe_temperature_to_phreeqc = bool(sync_temperature)
        self.validate_initial_gwe_fields = bool(validate_initial_fields)
        self.initial_gwe_field_tolerance = float(initial_field_tolerance)

    def _validate_feature_combinations(self) -> None:
        """Reject ownership conflicts before either native solver is run."""
        if self.sia_rate_evaluator is not None and (
            self.if_update_porosity_K
            or self.if_update_density
            or self.if_update_diffc
        ):
            raise ConfigurationError(
                "sia_rate_evaluator is a stateless aqueous-rate interface and "
                "cannot update PHREEQC-owned density, porosity, conductivity, "
                "or diffusion feedback"
            )
        if self.vsc_enabled and self.boundary_conductance_updates:
            raise ConfigurationError(
                "boundary_conductance_updates cannot be combined with VSC. "
                "MODFLOW VSC must remain the sole owner of viscosity-adjusted "
                "boundary and aquifer conductance."
            )
        if self.vsc_enabled and isinstance(
            self.perm_updater, FluidAdjustedKozenyCarmanUpdater
        ):
            raise ConfigurationError(
                "FluidAdjustedKozenyCarmanUpdater cannot be combined with VSC; "
                "that would apply viscosity to hydraulic conductivity twice"
            )

    def _init_state_containers(self) -> None:
        """
        Initialize runtime containers and backend placeholders.
        Parameters
        ----------
        None
            Uses instance configuration attributes.
        Returns
        -------
        None
            Creates empty containers for results and backend state.
        """
        self.phreeqc_rm = None
        self.modflow_api = None
        self.ncomps = None
        self.components = []
        self.headings = []
        self.is_setup = False
        self.results = []
        self.results_K = []
        self.results_porosity = []
        self.results_diffc = []
        self.results_temperature = []
        self.results_temperature_for_flow = []
        self.results_viscosity = []
        self.results_reference_K = []
        self.results_effective_K = []
        self.result_times = []
        self.modflow_convergence_failures = []
        self.sia_iterations = []
        self.sia_convergence_failures = []
        self.sia_diagnostics = []
        self.initial_concentrations = None
        self.selected_output = None
        self.energy_binding = None
        self.final_time_step_index = 0
        self.last_run_wall_time_seconds = None
        self.last_coupling_method = None
        self._run_active = False
        self._run_completed = False
        self._chemistry_finalized = False
        self._modflow_finalized = False

    def _init_fields(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        porosity: ArrayLike,
        saturation: ArrayLike,
        density: ArrayLike,
        viscosity: ArrayLike,
        d0: ArrayLike,
        porosity_update_mask: ArrayLike,
        print_chemistry_mask: ArrayLike,
        water_only_sink_rates: ArrayLike | None,
    ) -> None:
        """
        Initialize primary physical fields.
        Parameters
        ----------
        temperature, pressure, porosity, saturation, density, d0, print_chemistry_mask : ArrayLike
            Cell-wise fields for physical and chemical properties.
        Returns
        -------
        None
            Stores normalized arrays on the instance.
        """
        self.temperature = self._ensure_array("temperature", temperature)
        self.pressure = self._ensure_array("pressure", pressure)
        self.porosity = self._ensure_array("porosity", porosity)
        self.saturation = self._ensure_array("saturation", saturation)
        self.density = self._ensure_array("density", density)
        self.viscosity = self._ensure_array("viscosity", viscosity)
        self.porosity_update_mask = self._ensure_array(
            "porosity_update_mask", porosity_update_mask
        ).astype(bool)
        self.print_chemistry_mask = self._ensure_array(
            "print_chemistry_mask", print_chemistry_mask
        )
        if water_only_sink_rates is None:
            self.water_only_sink_rates = np.zeros(self.nxyz, dtype=float)
        else:
            self.water_only_sink_rates = self._ensure_array(
                "water_only_sink_rates", water_only_sink_rates
            )
            if np.any(self.water_only_sink_rates < 0.0):
                raise ValueError("water_only_sink_rates must be nonnegative")
        self.d0 = self._ensure_array("d0", d0)
        self.has_water_only_sinks = bool(np.any(self.water_only_sink_rates > 0.0))
        self._validate_physical_fields()

    def _validate_physical_fields(self) -> None:
        """Reject non-finite or physically impossible cell fields early."""
        fields = {
            "temperature": self.temperature,
            "pressure": self.pressure,
            "porosity": self.porosity,
            "saturation": self.saturation,
            "density": self.density,
            "viscosity": self.viscosity,
            "d0": self.d0,
            "water_only_sink_rates": self.water_only_sink_rates,
        }
        for name, values in fields.items():
            if not np.all(np.isfinite(values)):
                raise ConfigurationError(f"{name} contains non-finite values")
        if np.any(self.temperature <= -273.15):
            raise ConfigurationError("temperature must be above absolute zero")
        if np.any(self.pressure <= 0.0):
            raise ConfigurationError("pressure must be positive")
        if np.any(self.porosity <= 0.0) or np.any(self.porosity > 1.0):
            raise ConfigurationError("porosity must be in (0, 1]")
        if np.any(self.saturation < 0.0) or np.any(self.saturation > 1.0):
            raise ConfigurationError("saturation must be in [0, 1]")
        if np.any(self.density <= 0.0):
            raise ConfigurationError("density must be positive")
        if np.any(self.viscosity <= 0.0):
            raise ConfigurationError("viscosity must be positive")
        if np.any(self.d0 < 0.0):
            raise ConfigurationError("d0 must be nonnegative")

    def _ensure_array(self, name: str, value: ArrayLike) -> np.ndarray:
        """
        Normalize user input into a cell-wise array.
        Parameters
        ----------
        name : str
            Parameter name for error messages.
        value : ArrayLike
            Scalar or array input representing a field.
        Returns
        -------
        np.ndarray
            Flattened array with length nxyz.
        """
        return ensure_array(self.nxyz, name, value)

    def _initialize_phreeqcrm(self) -> None:
        """
        Initialize the PhreeqcRM backend.
        Parameters
        ----------
        None
            Uses instance configuration attributes.
        Returns
        -------
        None
            Creates and configures the PhreeqcRM object.
        """
        initialize_phreeqcrm(self)

    def _initialize_modflow6(self) -> None:
        """
        Initialize the MODFLOW 6 backend.
        Parameters
        ----------
        None
            Uses instance configuration attributes.
        Returns
        -------
        None
            Creates and configures the ModflowApi object.
        """
        initialize_modflow6(self)

    def _create_ic_array_from_map(self, ic_map: dict) -> np.ndarray:
        """
        Build the initial condition array for PhreeqcRM.
        Parameters
        ----------
        ic_map : dict
            Mapping of module name to initial condition values.
        Returns
        -------
        np.ndarray
            Packed initial condition array.
        """
        return create_ic_array_from_map(self.nxyz, ic_map)

    def _setup_single_ic(self, ic_map: dict) -> None:
        """
        Apply a single set of initial chemical conditions.
        Parameters
        ----------
        ic_map : dict
            Mapping of module name to initial condition values.
        Returns
        -------
        None
            Applies initial conditions to PhreeqcRM.
        """
        setup_single_ic(self.phreeqc_rm, self.nxyz, ic_map)

    def _setup_mixed_ic(self, ic_map1: dict, ic_map2: dict, fractions: ArrayLike) -> None:
        """
        Apply mixed initial chemical conditions.
        Parameters
        ----------
        ic_map1 : dict
            Mapping of module name to initial condition values.
        ic_map2 : dict
            Mapping of module name to initial condition values.
        fractions : ArrayLike
            Mixing fraction per cell for ic_map1.
        Returns
        -------
        None
            Applies mixed initial conditions to PhreeqcRM.
        """
        setup_mixed_ic(self.phreeqc_rm, self.nxyz, ic_map1, ic_map2, fractions)

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
            Optional mixing fraction for ic_map2.
        Returns
        -------
        np.ndarray
            Initial concentrations after equilibrium.
        """
        if self.is_setup:
            warnings.warn(
                "setup() has already completed; returning the cached initial concentrations",
                RuntimeWarning,
                stacklevel=2,
            )
            return self.initial_concentrations.copy()
        try:
            if ic_map2 is not None and fractions is not None:
                self._setup_mixed_ic(ic_map, ic_map2, fractions)
            elif ic_map2 is None and fractions is None:
                self._setup_single_ic(ic_map)
            else:
                raise ConfigurationError(
                    "ic_map2 and fractions must be provided together for mixed mode"
                )
            print("--- Running initial chemical equilibrium calculation ---")
            self.phreeqc_rm.SetTime(0.0 * SECONDS_PER_DAY)
            self.phreeqc_rm.SetTimeStep(0.0 * SECONDS_PER_DAY)
            self.phreeqc_rm.RunCells()
            initial = np.asarray(
                self.phreeqc_rm.GetConcentrations(), dtype=float
            ).ravel()
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
            raw_selected = np.asarray(
                self.phreeqc_rm.GetSelectedOutput(), dtype=float
            )
            if raw_selected.size != len(self.headings) * self.nxyz:
                raise CouplingError(
                    "Selected output size does not match headings and nxyz: "
                    f"{raw_selected.size} != {len(self.headings)} * {self.nxyz}"
                )
            self.selected_output = raw_selected.reshape(-1, self.nxyz)
            if not np.all(np.isfinite(self.selected_output)):
                raise CouplingError("Initial selected output contains non-finite values")
            if self.if_update_density:
                if self.use_phreeqc_calculated_density:
                    density = np.asarray(
                        self.phreeqc_rm.GetDensityCalculated(), dtype=float
                    ).ravel()
                    if density.size != self.nxyz or np.any(density <= 0.0):
                        raise CouplingError(
                            "PhreeqcRM calculated density is invalid or has the wrong size"
                        )
                    self.selected_output[-1] = density
                elif self.headings[-1].casefold() != self.density_output_heading.casefold():
                    raise ConfigurationError(
                        "Density feedback reads the final selected-output row. "
                        f"Expected heading {self.density_output_heading!r}, got "
                        f"{self.headings[-1]!r}. Set use_phreeqc_calculated_density=True "
                        "or provide the correct density_output_heading."
                    )
            if self.if_update_porosity_K:
                self._get_output_information()
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
        except Exception:
            self.is_setup = False
            raise

    def _get_output_information(self) -> None:
        """
        Extract mineral output indices and molar volumes.
        Parameters
        ----------
        None
            Uses instance headings and VM_minerals table.
        Returns
        -------
        None
            Stores indices and molar volumes on the instance.
        """
        output_indices, mineral_volumes, mineral_names = extract_output_information(
            self.headings, self.mineral_molar_volumes
        )
        self.output_indices = output_indices
        self.mineral_volumes = mineral_volumes
        self.d_mineral_names = mineral_names

    def _update_porosity(self) -> np.ndarray:
        """
        Update porosity using selected output mineral changes.
        Parameters
        ----------
        None
            Uses instance selected output and porosity fields.
        Returns
        -------
        np.ndarray
            Updated porosity field.
        """
        return update_porosity(
            self.selected_output, self.output_indices, self.mineral_volumes, self.porosity
        )

    def _update_K(
        self,
        K_old: np.ndarray,
        old_porosity: np.ndarray,
        new_porosity: np.ndarray,
        density_old: np.ndarray | None = None,
        density_new: np.ndarray | None = None,
        viscosity_old: np.ndarray | None = None,
        viscosity_new: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Update permeability using the configured updater.
        Parameters
        ----------
        K_old : np.ndarray
            Previous permeability field.
        old_porosity : np.ndarray
            Previous porosity field.
        new_porosity : np.ndarray
            Updated porosity field.
        Returns
        -------
        np.ndarray
            Updated permeability field.
        """
        return self.perm_updater.update(
            K_old,
            old_porosity,
            new_porosity,
            density_old=density_old,
            density_new=density_new,
            viscosity_old=viscosity_old,
            viscosity_new=viscosity_new,
        )

    def _update_diffc(self, new_porosity: np.ndarray) -> np.ndarray:
        """
        Update diffusion coefficient from porosity.
        Parameters
        ----------
        new_porosity : np.ndarray
            Updated porosity field.
        Returns
        -------
        np.ndarray
            Updated diffusion coefficient field.
        """
        return update_diffc(new_porosity, self.d0)

    def _get_species_slice(self, ispecies: int) -> slice:
        """
        Get slice for the specified component in a 1D vector.
        Parameters
        ----------
        ispecies : int
            Component index.
        Returns
        -------
        slice
            Slice representing the component block.
        """
        return get_species_slice(self.nxyz, ispecies)

    def run(self, method: CouplingMethod | str | None = None) -> None:
        """
        Run the standard reactive transport simulation loop.
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
            self._run_coupling(run_standard, CouplingMethod.SNIA)
            return
        normalized, runner = get_coupling_runner(method)
        self._run_coupling(runner, normalized)

    def run_SNIA(self) -> None:
        """Run the sequential non-iterative coupling loop explicitly."""
        self._run_coupling(run_standard, CouplingMethod.SNIA)

    def run_SIA(self) -> None:
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
        self._run_coupling(run_sia, CouplingMethod.SIA)

    def run_Strang(self) -> None:
        """Run symmetric transport-reaction-transport Strang splitting."""
        self._run_coupling(run_strang, CouplingMethod.STRANG)

    def run_ThermalSNIA(self) -> None:
        """Run explicit GWF-GWT-GWE-VSC reactive transport."""
        self._run_coupling(run_thermal_snia, CouplingMethod.THERMAL_SNIA)

    def _run_coupling(
        self, runner, method: CouplingMethod | str | None = None
    ) -> None:
        """Apply lifecycle guards around a coupling algorithm."""
        if self._run_active:
            raise CouplingError("A coupling run is already active")
        if self._run_completed:
            raise CouplingError(
                "This simulator has already completed a run; create a new instance "
                "for another simulation"
            )
        if (
            self.reaction_steps is not None
            and method is not CouplingMethod.SNIA
        ):
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
            raise ConfigurationError(
                "ThermalSNIA requires energy_enabled=True"
            )
        self._run_active = True
        if isinstance(method, CouplingMethod):
            self.last_coupling_method = method.value
        elif method is not None:
            self.last_coupling_method = str(method)
        else:
            self.last_coupling_method = runner.__name__
        try:
            runner(self)
        except Exception:
            raise
        else:
            self._run_completed = True
        finally:
            self._run_active = False

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
                "coupling_method": self.last_coupling_method,
                "logical_steps": self.final_time_step_index,
                "wall_time_seconds": self.last_run_wall_time_seconds,
                "modflow_convergence_failures": self.modflow_convergence_failures,
                "sia_iterations": self.sia_iterations,
                "sia_convergence_failures": self.sia_convergence_failures,
                "sia_diagnostics": self.sia_diagnostics,
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
        print("--- Finalizing simulation, releasing resources ---")
        if self.modflow_api is None:
            self._modflow_finalized = True
        elif not self._modflow_finalized:
            try:
                self.modflow_api.finalize()
                print("MODFLOW API closed")
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
            try:
                self.phreeqc_rm.CloseFiles()
                self.phreeqc_rm.MpiWorkerBreak()
                print("PhreeqcRM files closed.")
            except Exception as exc:
                warnings.warn(
                    f"PhreeqcRM finalization failed: {exc}",
                    ResourceWarning,
                    stacklevel=2,
                )
            finally:
                self._chemistry_finalized = True
        self.is_setup = False

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
        return list(self.phreeqc_rm.GetComponents())

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
        bc1 = np.full((1), number)
        return self.phreeqc_rm.InitialPhreeqc2Concentrations(bc1)


# PEP 8 public name for new code; the historical lowercase class remains the
# implementation so existing imports, notebooks, and serialized metadata keep
# working unchanged.
MF6PQC = mf6pqc
