import os
import time
import numbers
import numpy as np
import phreeqcrm
import modflowapi

from mf6pqc.permeability import KozenyCarmanUpdater, PowerLawUpdater

ArrayLike = int | float | list | tuple | np.ndarray

# todo
VM_minerals = {     # L/mol
    "Calcite":      0.03693,
    "Dolomite":     0.0645,
    "Halite":       0.0271,
    "Carnallite":   0.1737,
    "Polyhalite":   0.2180,
    "Sylvite":      0.0375,
    "Gypsum":       0.07421,
    "Bischofite":   0.1271,
    "Syngenite":    0.1273,
    "Ferrihydrite": 0.02399,
    "Jarosite":     0.15463,
    "Gibbsite":     0.03319,
    "Siderite":     0.02926,
}

class mf6pqc:
    """
    Reactive transport coupling MODFLOW 6 and PhreeqcRM
    """
    def __init__(
        self, 
        case_name: str = "temp_case", 

        nxyz:      int = 80, 
        nthreads:  int = 3, 

        temperature: ArrayLike = 25.0,
        pressure:    ArrayLike = 2.0,
        porosity:    ArrayLike = 0.35,
        saturation:  ArrayLike = 1.0,
        density:     ArrayLike = 1.0,
        d0:          ArrayLike = 1.0e-9*86400,

        print_chemistry_mask: ArrayLike = 1,

        componentH2O: bool = False,
        solution_density_volume: bool = False,

        db_path:          str = None, 
        pqi_path:         str = None, 
        modflow_dll_path: str = None, 
        output_dir:       str = None,
        workspace:        str = None,

        if_update_porosity_K: bool = False,
        if_update_density:    bool = False,
        if_update_diffc: bool = False,
    ):

        self.case_name = case_name
        self.nxyz = nxyz
        self.nthreads = nthreads

        self.componentH2O = componentH2O
        self.solution_density_volume = solution_density_volume
        
        self._temperature_in = temperature
        self._pressure_in = pressure
        self._porosity_in = porosity
        self._saturation_in = saturation
        self._density_in = density
        self.d0 = d0

        self._print_chemistry_mask_in = print_chemistry_mask

        self.db_path = db_path
        self.pqi_path = pqi_path
        self.modflow_dll_path = modflow_dll_path
        self.workspace = workspace
        self.output_dir = output_dir

        self.phreeqc_rm = None
        self.modflow_api = None
        self.ncomps = None # number of reactive components
        self.components = []
        self.headings = []

        self.is_setup = False
        self.if_update_density = if_update_density
        self.if_update_porosity_K = if_update_porosity_K
        self.if_update_diffc = if_update_diffc
        
        self.results = []
        self.results_K = []
        self.results_porosity = []
        self.results_diffc = []

        # initialization
        self._process_physical_properties()
        self._initialize_phreeqcrm()

        if self.if_update_porosity_K:
            self.perm_updater = KozenyCarmanUpdater() # PowerLawUpdater()
        
    def _set_physical_property(self, name: str, value: ArrayLike) -> np.ndarray:
        """
        General function to convert user provided scalars or arrays into a valid ndarray
        """
        if isinstance(value, numbers.Number):
            return np.full((self.nxyz,), float(value))
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float).ravel()
            if arr.shape != (self.nxyz,):
                raise ValueError(
                    f"The provided '{name}' list/array has length {arr.shape[0]},"
                    f"which does not match the total number of grid cells nxyz ({self.nxyz})."
                )
            return arr
        else:
            raise TypeError(f"'Unsupported type for parameter {name}', received {type(value).__name__}.")

    def _process_physical_properties(self):
        self.temperature = self._set_physical_property("temperature", self._temperature_in)
        self.pressure = self._set_physical_property("pressure", self._pressure_in)
        self.porosity = self._set_physical_property("porosity", self._porosity_in)
        self.saturation = self._set_physical_property("saturation", self._saturation_in)
        self.density = self._set_physical_property("density", self._density_in)
        # self.print_chemistry_mask = self._set_physical_property("mask list", self._print_chemistry_mask_in).astype(int)
        
    def _initialize_phreeqcrm(self):
        """
        Private method: initialize and configure the PhreeqcRM instance
        """
        print("--- Initializing PhreeqcRM ---")
        self.phreeqc_rm = phreeqcrm.PhreeqcRM(self.nxyz, self.nthreads)

        os.makedirs(self.output_dir, exist_ok=True)
        prefix = os.path.join(self.output_dir, f"{self.case_name}_prm")
        self.phreeqc_rm.SetFilePrefix(prefix)
        self.phreeqc_rm.OpenFiles()

        # todo Set units
        self.phreeqc_rm.SetUnitsSolution(2)     # ! Solution units: 1, mg/L; 2, mol/L; 3, kg/kgs
        self.phreeqc_rm.SetUnitsPPassemblage(0) # ! Mineral phase units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsExchange(0)     # ! Ion exchange units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsSurface(0)      # ! Surface complexation units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsGasPhase(0)     # ! Gas phase units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsSSassemblage(0) # ? Solid solution units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsKinetics(0)     # ! Kinetics units: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        
        # todo Time conversion
        self.phreeqc_rm.SetTimeConversion(1.0 / 86400.0)
        
        self.phreeqc_rm.SetTemperature(self.temperature)
        self.phreeqc_rm.SetPressure(self.pressure)
        self.phreeqc_rm.SetPorosity(self.porosity)
        self.phreeqc_rm.SetSaturation(self.saturation)
        self.phreeqc_rm.SetDensityUser(self.density) # todo
        # self.phreeqc_rm.SetRepresentativeVolume(np.ones((nxyz,)))

        self.phreeqc_rm.SetComponentH2O(self.componentH2O)
        self.phreeqc_rm.UseSolutionDensityVolume(self.solution_density_volume)
        # self.phreeqc_rm.SetPrintChemistryMask(self.print_chemistry_mask)

        print(f"Loading Phreeqc database: {self.db_path}")
        self.phreeqc_rm.LoadDatabase(self.db_path)
        
        # todo Enable printing of chemical processes
        self.phreeqc_rm.SetPrintChemistryOn(True, False, False)
        
        print(f"Running chemistry definition file: {self.pqi_path}")
        self.phreeqc_rm.RunFile(True, True, True, self.pqi_path)
        
        # ? Clear contents of worker and utility instances
        self.phreeqc_rm.RunString(True, False, True, "DELETE; -all")

        self.ncomps = self.phreeqc_rm.FindComponents()
        
        # Get the list of reactive components
        self.components = list(self.phreeqc_rm.GetComponents())
        print(f"List of reactive chemical components: {self.components}")
        
        self.phreeqc_rm.SetScreenOn(False)
        self.phreeqc_rm.SetSelectedOutputOn(True)
        
    def _initialize_modflow6(self):
        """
        Private method: initialize Modflow 6 instance
        """
        print(f"--- Initializing MODFLOW 6 ---")
        print(f"Working directory: {self.workspace}")
        try:
            self.modflow_api = modflowapi.ModflowApi(self.modflow_dll_path, working_directory=self.workspace)
        except Exception as e:
            print(f"Error: failed to load MODFLOW 6 DLL or set the working directory.")
            print(f"DLL path: {self.modflow_dll_path}")
            print(f"Working directory: {self.workspace}")
            raise e
        self.modflow_api.initialize()
        # load ApiSimulation object
        self.sim = modflowapi.extensions.ApiSimulation.load(self.modflow_api)

    def _create_ic_array_from_map(self, ic_map: dict) -> np.ndarray:
        """
        Private method: Create and return an initial condition array (ic_array) for PhreeqcRM based on the ic_map dictionary.
        """
        module_indices = {
                  'solution': 0,
        'equilibrium_phases': 1,
                  'exchange': 2, 
                   'surface': 3,
                 'gas_phase': 4,
           'solid_solutions': 5,
                  'kinetics': 6
        }
        ic_array = np.full((self.nxyz * 7,), -1, dtype=np.int32)
        for module_name, pqi_value in ic_map.items():
            if module_name not in module_indices:
                print(f"Warning: unknown chemical module name '{module_name}' will be ignored.")
                continue
            idx = module_indices[module_name]
            start, end = idx * self.nxyz, (idx + 1) * self.nxyz
            if isinstance(pqi_value, numbers.Number):
                ic_array[start:end] = int(pqi_value)
            elif isinstance(pqi_value, (list, tuple, np.ndarray)):
                arr = np.array(pqi_value, dtype=np.int32).ravel()
                if arr.shape != (self.nxyz,):
                    raise ValueError(f"The provided array for '{module_name}' does not match nxyz.")
                ic_array[start:end] = arr
            else:
                raise TypeError(f"'Unsupported parameter type: {module_name}'.")
        return ic_array

    def _setup_single_ic(self, ic_map: dict):
        """
        Private method: process single initial chemical condition
        """
        print("--- Setting single initial chemical condition ---")
        ic_array = self._create_ic_array_from_map(ic_map)
        self.phreeqc_rm.InitialPhreeqc2Module(ic_array)

    def _setup_mixed_ic(self, ic_map1: dict, ic_map2: dict, fractions: ArrayLike):
        """
        Private method: process mixed initial chemical conditions
        """
        print("--- Setting mixed initial chemical condition ---")
        ic_array1 = self._create_ic_array_from_map(ic_map1)
        ic_array2 = self._create_ic_array_from_map(ic_map2)
        
        # Process and validate the fractions input
        fraction_array = self._set_physical_property("mixing ratio", fractions)
        # Expand fractions to match PhreeqcRM requirements
        fractions_tiled = np.tile(fraction_array, 7)
        
        self.phreeqc_rm.InitialPhreeqc2Module_mix(ic_array1, ic_array2, fractions_tiled)

    def setup(
        self, 
        ic_map:         dict, 
        ic_map2:        dict | None = None, 
        fractions: ArrayLike | None = None
    ) -> np.ndarray:

        if self.is_setup:
            print("Warning: simulator has already been set up.")
            return
        self.is_setup = True

        # todo Dispatch to appropriate setup method
        if ic_map2 is not None and fractions is not None:
            self._setup_mixed_ic(ic_map, ic_map2, fractions) # Mixed initial condition
        elif ic_map2 is None and fractions is None:
            self._setup_single_ic(ic_map) # Single initial condition
        else:
            raise ValueError("Parameter mismatch: 'ic_map2' and 'fractions' must be provided together for mixed mode")

        # run initial equilibrium and obtain concentrations
        print("--- Running initial chemical equilibrium calculation ---")
        self.phreeqc_rm.SetTime(0.0)
        self.phreeqc_rm.SetTimeStep(0.0)
        self.phreeqc_rm.RunCells()
        initial_concentrations = self.phreeqc_rm.GetConcentrations()

        self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())
        if not self.headings:
            print("Warning: Failed to retrieve Selected Output headings.")

        self.selected_output = self.phreeqc_rm.GetSelectedOutput()
        self.selected_output = self.selected_output.reshape(-1, self.nxyz)
        self.results.append(self.selected_output)

        if self.if_update_porosity_K == True:
            self._get_output_information()

        return initial_concentrations
    
    def _get_output_information(self):
        """
        Private method: Extract various information from selected_output
        """

        output_indices = []
        mineral_volumes = []
        d_mineral_names = []

        for idx, heading in enumerate(self.headings):
            # Skip fields that do not start with d_ or are too short
            if not (heading.startswith("d_") and len(heading) > 2): # >2 avoids only "d_"
                continue

            # Extract mineral name
            d_mineral_name = heading[2:]
            d_mineral_names.append(d_mineral_name)
            output_indices.append(idx)

            # Get molar volume or raise error
            if d_mineral_name in VM_minerals:
                mineral_volumes.append(VM_minerals[d_mineral_name])
            else:
                raise ValueError(f"Cannot find molar volume for '{d_mineral_name}'")
            
        self.output_indices = np.array(output_indices, dtype=int)
        self.mineral_volumes = np.array(mineral_volumes, dtype=float).reshape(-1, 1)
        self.d_mineral_names = np.array(d_mineral_names)

    def _update_porosity(self) -> np.ndarray:

        mineral_delta_moles = self.selected_output[self.output_indices, :]
        # Compute total mineral volume change (mole change × molar volume), L/mol
        total_volume_change = np.sum(self.mineral_volumes * mineral_delta_moles, axis=0)
        new_porosity = self.porosity - total_volume_change
        
        new_porosity = np.maximum(1e-20, new_porosity)
        new_porosity = np.minimum(1.0,   new_porosity)

        return new_porosity

    def _update_K(self, K_old: np.ndarray, old_porosity: np.ndarray, new_porosity: np.ndarray):
        return self.perm_updater.update(K_old, old_porosity, new_porosity)
    
    def _update_diffc(self, new_porosity: np.ndarray) -> np.ndarray:
        # Compute tortuosity factor τ (tau)
        tortuosity_factor = new_porosity ** (1.0 / 3.0)
        # Compute effective diffusion coefficient De
        diffc = tortuosity_factor * self.d0
        return diffc

    def _get_species_slice(self, ispecies: int) -> slice:
        """
        todo Return slice for the ispecies-th solute in the 1D concentration vector
        """
        return slice(ispecies * self.nxyz, (ispecies + 1) * self.nxyz)

    def run(self):
        """
        Main loop for reactive transport simulation
        """
        self._initialize_modflow6()
        if not self.is_setup:
            raise RuntimeError("setup() must be called before running the simulation")
        
        print("\n--- Starting reactive transport simulation ---")
        start_sim_time = time.time()

        # --- Cache MODFLOW 6 variable addresses and shapes ---
        conc_var_info = {}
        for sp_name in self.components:
            gwt_model_name = f"gwt_{sp_name}_model"
            address = self.modflow_api.get_var_address("X", gwt_model_name)

            ptr = self.modflow_api.get_value_ptr(address)
            shape = ptr.shape
            conc_var_info[sp_name] = {
                "address": address,
                "ptr": ptr,
                "shape": shape
            }
            print(f"  - solute '{sp_name}', shape={shape}")

        if self.if_update_porosity_K:
            self.K11_tag = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
            self.K33_tag = self.modflow_api.get_var_address("K33", "gwf_model", "NPF")
            self.KCHANGEPER_tag = self.modflow_api.get_var_address("KCHANGEPER", "gwf_model", "NPF")
            self.KCHANGESTP_tag = self.modflow_api.get_var_address("KCHANGESTP", "gwf_model", "NPF")
            self.NODEKCHANGE_tag = self.modflow_api.get_var_address("NODEKCHANGE", "gwf_model", "NPF")

            # Cache GWT thetam / porosity addresses
            self.thetam_tags = {
                sp: self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
                for sp in self.components
            }

            # todo !
            self.modflow_api.set_value(self.KCHANGEPER_tag, np.array([self.sim.kper+2]))
            self.modflow_api.set_value(self.NODEKCHANGE_tag, np.ones(self.nxyz, dtype=np.int32))
            
            current_K11 = self.modflow_api.get_value(self.K11_tag).copy()
            
            self.results_porosity = [self.porosity.copy()]
            self.results_K = [current_K11.copy()]

        if self.if_update_diffc:
            # Cache GWT thetam / porosity addresses
            self.thetam_tags = {
                sp: self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
                for sp in self.components
            }
            self.diffc_tags = {
                sp: self.modflow_api.get_var_address("DIFFC", f"gwt_{sp}_model", "DSP")
                for sp in self.components
            }

        # IMS for XMI
        # ims_pkg_addr = self.modflow_api.get_var_address("MXITER", "SLN_1")
        # max_iter = self.modflow_api.get_value_ptr(ims_pkg_addr)[0]
        # density_address = self.modflow_api.get_var_address("DENSE", "gwf_model", "BUY")
        # n_solutions = self.modflow_api.get_subcomponent_count()

        # Preallocate large arrays (avoid realloc each timestep)
        conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float)
        conc_after_reaction = np.empty_like(conc_from_mf)

        species_slices = [self._get_species_slice(i) for i in range(self.ncomps)]

        # selected output
        temp_selected = self.phreeqc_rm.GetSelectedOutput()
        nsel = len(temp_selected) // self.nxyz
        
        # initial state
        self.selected_output = temp_selected.reshape(-1, self.nxyz)
        self.results.append(self.selected_output.copy())

        current_time = self.modflow_api.get_current_time()
        end_time = self.modflow_api.get_end_time()
        time_step_index = 0

        # todo temp
        mf_time = 0.0
        phreeqc_time = 0.0
        copy_time = 0.0
        qita_time = 0.0

        while current_time < end_time:

            # --- Step 1: Update MODFLOW 6 ---
            t0 = time.time()
            dt = self.modflow_api.get_time_step()

            # ! --- BMI ---
            self.modflow_api.update()

            mf_time += time.time() - t0

            # ! --- XMI solve Start --- #
            # self.modflow_api.prepare_time_step(dt)

            # # aaa = self.phreeqc_rm.GetDensityCalculated()
            # # self.modflow_api.set_value(density_address, aaa*1000.0)

            # for solution_id in range(1, n_solutions + 1):
            #     self.modflow_api.prepare_solve(solution_id)
            #     aaa = self.phreeqc_rm.GetDensityCalculated()
            #     self.modflow_api.set_value(density_address, aaa*1000.0)
            #     kiter = 0
            #     has_converged = False
            #     while kiter < max_iter:
            #         has_converged = self.modflow_api.solve(solution_id)
            #         kiter += 1
            #         if has_converged:
            #             break
            
            #     self.modflow_api.finalize_solve(solution_id)
            #     self.modflow_api.finalize_time_step()
            # ! --- XMI solve End --- #

            current_time = self.modflow_api.get_current_time()

            # --- Step 2: Acquire X from MODFLOW 6 ---
            t0 = time.time()
            for i, sp in enumerate(self.components):
                ptr = conc_var_info[sp]["ptr"]
                sl = species_slices[i]
                conc_from_mf[sl] = ptr.reshape(-1)
            copy_time += (time.time() - t0)
                
            # --- Step 3: Update PhreeqcRM ---
            t0 = time.time()
            self.phreeqc_rm.SetConcentrations(conc_from_mf)
            self.phreeqc_rm.SetTime(current_time)
            self.phreeqc_rm.SetTimeStep(dt)
            self.phreeqc_rm.RunCells()
            phreeqc_time += time.time() - t0

            conc_after_reaction[:] = self.phreeqc_rm.GetConcentrations()
            
            # --- Step 4: Write reacted concentrations back to MODFLOW ---
            t0 = time.time()
            for i, sp in enumerate(self.components):
                ptr = conc_var_info[sp]["ptr"]
                sl = species_slices[i]
                ptr[:] = conc_after_reaction[sl].reshape(ptr.shape)
            copy_time += (time.time() - t0)

            # --- Step 5: Update porosity / K ---
            t0 = time.time()
            if self.if_update_porosity_K:
                old_porosity = self.porosity
                new_porosity = self._update_porosity()
                self.porosity = new_porosity
                
                # Update PhreeqcRM porosity
                self.phreeqc_rm.SetPorosity(self.porosity)
                
                # Update GWT porosity / thetam
                for sp in self.components:
                    self.modflow_api.set_value(self.thetam_tags[sp], new_porosity)

                self.results_porosity.append(new_porosity.copy())

                # ------------------

                current_K11 = self._update_K(current_K11, old_porosity, new_porosity)
                self.results_K.append(current_K11.copy())

                self.modflow_api.set_value(self.KCHANGESTP_tag, np.array([self.sim.kstp+2]))

                self.modflow_api.set_value(self.K11_tag, current_K11)
                self.modflow_api.set_value(self.K33_tag, current_K11 * 0.1)

            if self.if_update_diffc:
                # Update MODFLOW diffusion coefficient DIFFC
                new_diffc = self._update_diffc(new_porosity)
                self.results_diffc.append(new_diffc.copy())

                for sp in self.components:
                    self.modflow_api.set_value(self.diffc_tags[sp], new_diffc)
            qita_time += (time.time() - t0)
            
            # --- Save selected output ---
            temp_selected = self.phreeqc_rm.GetSelectedOutput()
            self.selected_output = temp_selected.reshape(nsel, self.nxyz)
            self.results.append(self.selected_output.copy())

            time_step_index += 1

            # Print every 10 steps
            if time_step_index % 1 == 0:
                print(f"  t = {current_time:.2f}/{end_time:.2f} days, step={time_step_index}")

        # todo !
        print("\n--- Timing Summary ---")
        print(f"MODFLOW time:      {mf_time:.3f} s")
        print(f"PHREEQCRM time:    {phreeqc_time:.3f} s")
        print(f"Copy/reshape time: {copy_time:.3f} s")
        print(f"Others:            {qita_time:.3f} s")

        print(f"Total time: {mf_time + phreeqc_time + copy_time + qita_time:.3f} s")

        self.results = np.array(self.results)

        if self.if_update_porosity_K:
            self.results_porosity = np.array(self.results_porosity)
            self.results_K = np.array(self.results_K)

        if self.if_update_diffc:
            self.results_diffc = np.array(self.results_diffc)

        self.final_time_step_index = time_step_index

        print(f"--- Simulation finished, steps={time_step_index}, time={time.time()-start_sim_time:.2f} s ---")

    # todo 
    def run_with_SIA(self):
        """
        Main loop for the coupled simulation (SIA / Picard Iteration)
        """
        self._initialize_modflow6()
        if not self.is_setup:
            raise RuntimeError("setup() must be called before running the simulation")
        
        print("\n--- Starting SIA/Picard ---")
        start_sim_time = time.time()
        
        conc_var_info = {}
        for sp_name in self.components:
            gwt_model_name = f"gwt_{sp_name}_model"
            address = self.modflow_api.get_var_address("X", gwt_model_name)
            ptr = self.modflow_api.get_value_ptr(address)
            conc_var_info[sp_name] = {
                "address": address, "ptr": ptr, "shape": ptr.shape
            }

        if self.if_update_porosity_K:
            self.K11_tag = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
            self.K33_tag = self.modflow_api.get_var_address("K33", "gwf_model", "NPF")

            self.KCHANGEPER_tag = self.modflow_api.get_var_address("KCHANGEPER", "gwf_model", "NPF")
            self.KCHANGESTP_tag = self.modflow_api.get_var_address("KCHANGESTP", "gwf_model", "NPF")
            self.NODEKCHANGE_tag = self.modflow_api.get_var_address("NODEKCHANGE", "gwf_model", "NPF")

            self.thetam_tags = {
                sp: self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
                for sp in self.components
            }

            self.modflow_api.set_value(self.KCHANGEPER_tag, np.array([self.sim.kper+2]))
            self.modflow_api.set_value(self.NODEKCHANGE_tag, np.ones(self.nxyz, dtype=np.int32))
            
            current_K11 = self.modflow_api.get_value(self.K11_tag).copy()
            
            self.results_porosity = [self.porosity.copy()]
            self.results_K = [current_K11.copy()]

        if self.if_update_diffc:
            self.diffc_tags = {
                sp: self.modflow_api.get_var_address("DIFFC", f"gwt_{sp}_model", "DSP")
                for sp in self.components
            }
            self.results_diffc = []

        # todo
        ims_addr = self.modflow_api.get_var_address("MXITER", "SLN_1")
        max_inner_iter = self.modflow_api.get_value_ptr(ims_addr)[0]
        n_solutions = self.modflow_api.get_subcomponent_count()
        
        conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float)
        conc_after_reaction = np.empty_like(conc_from_mf)
        species_slices = [self._get_species_slice(i) for i in range(self.ncomps)]

        temp_selected = self.phreeqc_rm.GetSelectedOutput()
        nsel = len(temp_selected) // self.nxyz
        self.selected_output = temp_selected.reshape(-1, self.nxyz)
        self.results = [self.selected_output.copy()]

        # Picard iteration parameters
        max_picard_iter = 100
        picard_tol = 1e-4     # convergence tolerance
        
        current_time = self.modflow_api.get_current_time()
        end_time = self.modflow_api.get_end_time()
        time_step_index = 0

        while current_time < end_time:

            dt = self.modflow_api.get_time_step()

            self.modflow_api.prepare_time_step(dt)

            picard_k = 0
            picard_converged = False

            prev_iter_porosity = self.porosity.copy()

            # Picard Loop
            while picard_k < max_picard_iter:
                if self.if_update_porosity_K:
                    self.modflow_api.set_value(self.KCHANGESTP_tag, np.array([self.sim.kstp+2]))
                
                for solution_id in range(1, n_solutions + 1):
                    self.modflow_api.prepare_solve(solution_id)
                    
                    # MODFLOW Inner Loop
                    kiter = 0
                    has_converged = False
                    while kiter < max_inner_iter:
                        has_converged = self.modflow_api.solve(solution_id)
                        kiter += 1
                        if has_converged:
                            break

                    self.modflow_api.finalize_solve(solution_id)

                for i, sp in enumerate(self.components):
                    ptr = conc_var_info[sp]["ptr"]
                    sl = species_slices[i]
                    conc_from_mf[sl] = ptr.reshape(-1)

                self.phreeqc_rm.SetConcentrations(conc_from_mf)
                self.phreeqc_rm.SetTime(current_time)
                self.phreeqc_rm.SetTimeStep(dt)
                self.phreeqc_rm.RunCells()

                conc_after_reaction[:] = self.phreeqc_rm.GetConcentrations()

                for i, sp in enumerate(self.components):
                    ptr = conc_var_info[sp]["ptr"]
                    sl = species_slices[i]
                    ptr[:] = conc_after_reaction[sl].reshape(ptr.shape)

                if self.if_update_porosity_K:
                    new_porosity_iter = self._update_porosity()
                    max_change = np.max(np.abs(new_porosity_iter - prev_iter_porosity))
                    
                    self.porosity = new_porosity_iter
                    prev_iter_porosity = new_porosity_iter.copy()

                    current_K11 = self._update_K(current_K11, self.porosity, new_porosity_iter) 

                    self.modflow_api.set_value(self.K11_tag, current_K11)
                    self.modflow_api.set_value(self.K33_tag, current_K11 * 0.1)

                    for sp in self.components:
                        self.modflow_api.set_value(self.thetam_tags[sp], self.porosity)

                    self.phreeqc_rm.SetPorosity(self.porosity)

                    if self.if_update_diffc:
                        new_diffc = self._update_diffc(self.porosity)
                        for sp in self.components:
                            self.modflow_api.set_value(self.diffc_tags[sp], new_diffc)

                    if max_change < picard_tol:
                        picard_converged = True
                        print(f"  Picard converged at iteration {picard_k+1}, Max Delta Porosity: {max_change:.2e}")
                        break
                else:
                    picard_converged = True
                    break

                picard_k += 1
            
            if not picard_converged:
                print(f"Warning: Time step {time_step_index} Picard did NOT converge (Iter={max_picard_iter})")

            self.modflow_api.finalize_time_step()
            current_time = self.modflow_api.get_current_time()

            temp_selected = self.phreeqc_rm.GetSelectedOutput()
            self.selected_output = temp_selected.reshape(nsel, self.nxyz)
            self.results.append(self.selected_output.copy())

            if self.if_update_porosity_K:
                self.results_porosity.append(self.porosity.copy())
                self.results_K.append(current_K11.copy())
            
            if self.if_update_diffc and 'new_diffc' in locals():
                self.results_diffc.append(new_diffc.copy())

            time_step_index += 1
            if time_step_index % 10 == 0:
                print(f"  t = {current_time:.2f}/{end_time:.2f}, step={time_step_index}, Picard Iters={picard_k+1}")

        self.results = np.array(self.results)
        if self.if_update_porosity_K:
            self.results_porosity = np.array(self.results_porosity)
            self.results_K = np.array(self.results_K)
        if self.if_update_diffc:
            self.results_diffc = np.array(self.results_diffc)

        self.final_time_step_index = time_step_index
        print(f"--- Simulation finished, steps={time_step_index}, elapsed={time.time()-start_sim_time:.2f} sec ---")

    def save_results(self, filename: str = None):
        """
        todo Save selected outputs, porosity, and permeability results into .npy files.
        """
        if not hasattr(self, 'final_time_step_index') or self.final_time_step_index == 0:
            print("Warning: no valid simulation results to save")
            return

        if filename is None:
            # filename = os.path.join(self.output_dir, f"{self.case_name}_results.npy")
            filename = os.path.join(self.output_dir, f"results.npy")
        else:
            filename = os.path.join(self.output_dir, filename)
        base = os.path.splitext(filename)[0]

        actual_data_points = self.final_time_step_index + 1

        if self.headings:
            np.save(filename, self.results[:actual_data_points])
            print(f"Results saved to: {filename} ({actual_data_points} time points)")
            
            header_file = base + "_headings.txt"
            with open(header_file, 'w') as f:
                for heading in self.headings:
                    f.write(f"{heading}\n")
            print(f"Headings saved to: {header_file}")

            if self.if_update_porosity_K:
                porosity_file = base + "_porosity.npy"
                np.save(porosity_file, self.results_porosity[:actual_data_points])
                print(f"Porosity results saved to: {porosity_file}")

                k_file = base + "_K.npy"
                np.save(k_file, self.results_K[:actual_data_points])
                print(f"K results saved to: {k_file}")

            if self.if_update_diffc:
                diffc_file = base + "_diffc.npy"
                np.save(diffc_file, self.results_diffc) 
                print(f"DIFFC results saved to: {diffc_file}")
        else:
            print("Error: Cannot obtain headings, result dimension unknown")

    def finalize(self):
        """
        Finalize simulation, close files and free resources.
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

    def get_components(self):
        return list(self.phreeqc_rm.GetComponents())
    
    def get_initial_concentrations(self, number):
        """Retrieve boundary concentration for a single value."""
        bc1 = np.full((1), number)
        #phreeqc_rm.InitialPhreeqc2Concentrations(bc_conc_dbl_vect, bc1)
        return self.phreeqc_rm.InitialPhreeqc2Concentrations(bc1)