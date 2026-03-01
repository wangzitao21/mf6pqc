import numpy as np

from mf6pqc.types import ArrayLike
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
from mf6pqc.coupling_logic import (
    initialize_phreeqcrm,
    initialize_modflow6,
    run_standard,
    run_sia,
)
from mf6pqc.permeability import KozenyCarmanUpdater, PowerLawUpdater

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
    ):
        """
        Initialize a coupled MODFLOW 6 and PhreeqcRM simulator.
        Parameters
        ----------
        See class signature for configuration options.
        """
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
        )
        self._set_runtime_flags(if_update_porosity_K, if_update_density, if_update_diffc)
        self._init_state_containers()
        self._init_fields(temperature, pressure, porosity, saturation, density, d0, print_chemistry_mask)
        self._initialize_phreeqcrm()
        if self.if_update_porosity_K:
            self.perm_updater = KozenyCarmanUpdater()

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
        self.case_name = case_name
        self.nxyz = nxyz
        self.nthreads = nthreads
        self.componentH2O = componentH2O
        self.solution_density_volume = solution_density_volume
        self.db_path = db_path
        self.pqi_path = pqi_path
        self.modflow_dll_path = modflow_dll_path
        self.output_dir = output_dir
        self.workspace = workspace
        self.save_interval = save_interval

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

    def _init_fields(
        self,
        temperature: ArrayLike,
        pressure: ArrayLike,
        porosity: ArrayLike,
        saturation: ArrayLike,
        density: ArrayLike,
        d0: ArrayLike,
        print_chemistry_mask: ArrayLike,
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
        self.print_chemistry_mask = self._ensure_array(
            "print_chemistry_mask", print_chemistry_mask
        )
        self.d0 = self._ensure_array("d0", d0)

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
            print("Warning: simulator has already been set up.")
            return
        self.is_setup = True
        if ic_map2 is not None and fractions is not None:
            self._setup_mixed_ic(ic_map, ic_map2, fractions)
        elif ic_map2 is None and fractions is None:
            self._setup_single_ic(ic_map)
        else:
            raise ValueError(
                "Parameter mismatch: 'ic_map2' and 'fractions' must be provided together for mixed mode"
            )
        print("--- Running initial chemical equilibrium calculation ---")
        self.phreeqc_rm.SetTime(0.0 * SECONDS_PER_DAY)
        self.phreeqc_rm.SetTimeStep(0.0 * SECONDS_PER_DAY)
        self.phreeqc_rm.RunCells()
        initial_concentrations = self.phreeqc_rm.GetConcentrations()
        self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())
        if not self.headings:
            print("Warning: Failed to retrieve Selected Output headings.")
        temp_selected = self.phreeqc_rm.GetSelectedOutput()
        self.selected_output = temp_selected.reshape(-1, self.nxyz)
        self.results.append(self.selected_output)
        if self.if_update_porosity_K:
            self._get_output_information()
        return initial_concentrations

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
            self.headings, VM_minerals
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
        self, K_old: np.ndarray, old_porosity: np.ndarray, new_porosity: np.ndarray
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
        return self.perm_updater.update(K_old, old_porosity, new_porosity)

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

    def run(self) -> None:
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
        run_standard(self)

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
        run_sia(self)

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
        print("--- Finalizing simulation, releasing resources ---")
        if self.modflow_api:
            self.modflow_api.finalize()
            print("MODFLOW API closed")
        if self.phreeqc_rm:
            self.phreeqc_rm.CloseFiles()
            self.phreeqc_rm.MpiWorkerBreak()
            print("PhreeqcRM files closed.")
        self.is_setup = False

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
