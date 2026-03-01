import os
import time
import numbers
import numpy as np
import phreeqcrm
import modflowapi
from scipy.ndimage import median_filter

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

        save_interval: int = 1,
    ):

        self.case_name = case_name
        self.nxyz = nxyz
        self.nthreads = nthreads

        self.componentH2O = componentH2O
        self.solution_density_volume = solution_density_volume

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

        self.save_interval = save_interval
        
        # initialization
        self.temperature = self._ensure_array("temperature", temperature)
        self.pressure    = self._ensure_array("pressure", pressure)
        self.porosity    = self._ensure_array("porosity", porosity)
        self.saturation  = self._ensure_array("saturation", saturation)
        self.density     = self._ensure_array("density", density)
        self.print_chemistry_mask = self._ensure_array("print_chemistry_mask", print_chemistry_mask)
        self.d0 = self._ensure_array("d0", d0)

        self._initialize_phreeqcrm()

        if self.if_update_porosity_K:
            self.perm_updater = KozenyCarmanUpdater() # PowerLawUpdater()

    def _ensure_array(self, name: str, value: ArrayLike) -> np.ndarray:
        """
        General function to convert user provided scalars or arrays into a valid ndarray
        """
        if isinstance(value, numbers.Number):
            return np.full((self.nxyz,), float(value))
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float).ravel()
            if arr.shape != (self.nxyz,):
                raise ValueError(
                    f"Parameter '{name}' length {arr.shape[0]} does not match nxyz ({self.nxyz})."
                )
            return arr
        else:
            raise TypeError(f"Unsupported type for {name}: {type(value).__name__}")
        
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
        self.phreeqc_rm.SetTimeConversion(1.0) # 1.0 / 86400.0
        
        self.phreeqc_rm.SetTemperature(self.temperature)
        self.phreeqc_rm.SetPressure(self.pressure)
        self.phreeqc_rm.SetPorosity(self.porosity)
        self.phreeqc_rm.SetSaturation(self.saturation)

        # self.phreeqc_rm.SetDensityUser(self.density) # 使用 mol/L 时闲置
        # self.phreeqc_rm.SetRepresentativeVolume(np.ones((nxyz,)))

        self.phreeqc_rm.SetComponentH2O(self.componentH2O)
        self.phreeqc_rm.UseSolutionDensityVolume(self.solution_density_volume)
        # self.phreeqc_rm.SetPrintChemistryMask(self.print_chemistry_mask)

        # 开启自动负载均衡 0.5 表示：如果不同线程的计算时间差异超过 50%，就重新分配网格
        self.phreeqc_rm.SetRebalanceFraction(0.5)
        
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
        fraction_array = self._ensure_array("mixing ratio", fractions)
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
        self.phreeqc_rm.SetTime(0.0 * 86400.0)
        self.phreeqc_rm.SetTimeStep(0.0 * 86400.0)
        self.phreeqc_rm.RunCells()
        initial_concentrations = self.phreeqc_rm.GetConcentrations()

        self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())
        if not self.headings:
            print("Warning: Failed to retrieve Selected Output headings.")

        # ! 保存初始状态
        temp_selected = self.phreeqc_rm.GetSelectedOutput()
        self.selected_output = temp_selected.reshape(-1, self.nxyz)
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
        
        new_porosity = np.maximum(1e-4, new_porosity)
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

        # ====== 潜水处理 START ====== #
        self.head_addr = self.modflow_api.get_var_address("X", "gwf_model")

        self.botm_arr = self.sim.get_model("gwf_model").dis.bot.values.ravel()
        self.top_arr = self.sim.get_model("gwf_model").dis.top.values.ravel()

        self.cell_thick = self.top_arr - self.botm_arr
        # botm_addr = self.modflow_api.get_var_address("BOTM", "DIS")
        # top_addr  = self.modflow_api.get_var_address("TOP",  "DIS")
        # idomain_addr = self.modflow_api.get_var_address("IDOMAIN", "DIS")
        
        # 读取数据
        # self.botm_arr = self.sim.get_model("gwf_model").dis.bot.values
        # self.modflow_api.get_value(botm_addr).ravel() # 展平为 1D
        # model_top = self.modflow_api.get_value(top_addr)              # 通常是 2D (nrow, ncol)
        # self.idomain = self.modflow_api.get_value(idomain_addr).ravel()

        # --- 构建每一层的 cell_tops 数组 ---
        # 逻辑：Layer 1 的 Top 是 model_top；Layer n 的 Top 是 Layer n-1 的 Botm
        # 我们需要知道 nlay, nrow, ncol 来正确处理
        # 这里的处理稍微依赖于 MODFLOW 数据的原始形状 (nlay, nrow, ncol)
        
        # 获取原始形状
        # ptr_botm = self.modflow_api.get_value_ptr(botm_addr)

        # nlay, nrow, ncol = ptr_botm.shape
        # self.cell_tops = np.zeros_like(self.botm_arr)
        
        # 展平以便操作
        # flat_botm = ptr_botm.reshape(nlay, -1) # (nlay, nc*nr)
        # flat_top  = model_top.reshape(1, -1)   # (1,    nc*nr)
        
        # 构造所有层的 Top
        # Layer 1 top = model top
        # Layer 2..n top = Layer 1..n-1 bottom
        # all_tops_list = []
        # all_tops_list.append(flat_top[0]) # Layer 1
        # for k in range(nlay - 1):
            # all_tops_list.append(flat_botm[k]) # Layer k+1 的顶是 k 的底
        
        # self.cell_tops = np.concatenate(all_tops_list).ravel()

        # 计算单元厚度 (用于分母，防止除以0)
        
        # self.cell_thick[self.cell_thick <= 0] = 1.0 # 避免除零错误 (对非活动单元)
        # ====== 潜水处理 END   ====== #

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

        # 获取各 GWT 的孔隙度
        self.thetam_ptrs = {}
        for sp in self.components:
            thetam_addr = self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
            self.thetam_ptrs[sp] = self.modflow_api.get_value_ptr(thetam_addr)

        if self.if_update_porosity_K:

            self.K11_addr = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
            self.K33_addr = self.modflow_api.get_var_address("K33", "gwf_model", "NPF")

            self.K11_ptr = self.modflow_api.get_value_ptr(self.K11_addr)
            self.K33_ptr = self.modflow_api.get_value_ptr(self.K33_addr)

            current_K11 = self.K11_ptr.copy()

            self.tdis_kper_addr = self.modflow_api.get_var_address("KPER", "TDIS")
            self.tdis_kstp_addr = self.modflow_api.get_var_address("KSTP", "TDIS")

            self.kchangeper_addr  = self.modflow_api.get_var_address("KCHANGEPER", "gwf_model", "NPF")
            self.kchangestp_addr  = self.modflow_api.get_var_address("KCHANGESTP", "gwf_model", "NPF")
            self.nodekchange_addr = self.modflow_api.get_var_address("NODEKCHANGE", "gwf_model", "NPF")
            
            self.modflow_api.set_value(self.nodekchange_addr, np.ones(self.nxyz, dtype=np.int32))

            # Cache GWT thetam / porosity addresses
            # self.thetam_tags = {
            #     sp: self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
            #     for sp in self.components
            # }
            
            self.results_porosity = [self.porosity.copy()]
            self.results_K = [current_K11]

        # ! 用于更新扩散系数 ===================== 
        if self.if_update_diffc:
            # Cache GWT thetam / porosity addresses
            # self.thetam_tags = {
            #     sp: self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
            #     for sp in self.components
            # }
            self.diffc_tags = {
                sp: self.modflow_api.get_var_address("DIFFC", f"gwt_{sp}_model", "DSP")
                for sp in self.components
            }
        # ! 用于更新扩散系数 =====================

        if self.if_update_density:
            self.density_addr = self.modflow_api.get_var_address("DENSE", "gwf_model", "BUY")
            self.density_ptr = self.modflow_api.get_value_ptr(self.density_addr)

        # ! 可用于更新蒸发量 ===================== START =====================
        # rch_addr = self.modflow_api.get_var_address("RECHARGE", "gwf_model", "RCH")
        # recharge_array = self.modflow_api.get_value_ptr(rch_addr)
        # print("蒸发量: ", self.modflow_api.get_value_ptr(rch_addr))
        # ! 可用于更新蒸发量 =====================  END  =====================

        # ======================= IMS for XMI ======================= # 
        n_solutions = self.modflow_api.get_subcomponent_count()
        # todo 这里硬编码了 "SLN_1"。意味着只读取了 GWF 模型 的最大迭代次数（比如 50）
        # ims_pkg_addr = self.modflow_api.get_var_address("MXITER", "SLN_1")
        # max_iter = self.modflow_api.get_value_ptr(ims_pkg_addr)[0]
        # 改进上述问题
        sol_iters_map = {}
        for solution_id in range(1, n_solutions + 1):
            sln_name = f"SLN_{solution_id}"
            ims_addr = self.modflow_api.get_var_address("MXITER", sln_name)
            # get_value_ptr 我们存数组本身，读取时再取 [0]）
            sol_iters_map[solution_id] = self.modflow_api.get_value_ptr(ims_addr)
            # print(f"  - Cached MXITER for {sln_name}")
        # ======================= IMS for XMI ======================= # 

        # Preallocate large arrays (avoid realloc each timestep)
        conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float)
        conc_after_reaction = np.empty_like(conc_from_mf)

        species_slices = [self._get_species_slice(i) for i in range(self.ncomps)]


        current_time = self.modflow_api.get_current_time()
        end_time = self.modflow_api.get_end_time()
        time_step_index = 0

        # self.phreeqc_rm.SetSaturation(np.ones(self.nxyz)*(0.0))
        while current_time < end_time:

            # --- Step 1: Update MODFLOW 6 ---
            t0 = time.time()
            dt = self.modflow_api.get_time_step()

            # ! --- XMI solve Start --- #
            self.modflow_api.prepare_time_step(dt)

            if self.if_update_porosity_K and time_step_index > 0:
                
                current_tdis_kper = self.modflow_api.get_value(self.tdis_kper_addr)
                current_tdis_kstp = self.modflow_api.get_value(self.tdis_kstp_addr)

                self.modflow_api.set_value(self.kchangeper_addr, current_tdis_kper)
                self.modflow_api.set_value(self.kchangestp_addr, current_tdis_kstp)

                self.K11_ptr[:] = current_K11
                self.K33_ptr[:] = current_K11 * 0.1

            # ! --- BMI ---
            # self.modflow_api.update()

            if self.if_update_density:
                # current_density = self.phreeqc_rm.GetDensityCalculated() * 1000
                current_density = self.phreeqc_rm.GetSelectedOutput()[-self.nxyz:]* 1000 # ! 权衡之举

            for solution_id in range(1, n_solutions + 1):
                self.modflow_api.prepare_solve(solution_id)

                if (self.if_update_density == True) and (solution_id == 1):
                    self.density_ptr[:] = current_density

                current_max_iter = sol_iters_map[solution_id][0]
                kiter = 0
                has_converged = False
                # while kiter < max_iter:
                while kiter < current_max_iter:
                    has_converged = self.modflow_api.solve(solution_id)
                    kiter += 1
                    if has_converged:
                        break
            
                self.modflow_api.finalize_solve(solution_id)

            self.modflow_api.finalize_time_step()

            # ! --- XMI solve End --- #
            current_time = self.modflow_api.get_current_time()

            # --- Step 2: Acquire X from MODFLOW 6 ---
            for i, sp in enumerate(self.components):
                ptr = conc_var_info[sp]["ptr"]
                sl = species_slices[i]
                conc_from_mf[sl] = ptr #.reshape(-1)
                
            conc_from_mf[conc_from_mf < 1e-20] = 1e-20
            
            # --- Step 3: Update PhreeqcRM ---

            # ! ====== 潜水处理 START ====== #
            # current_head = self.modflow_api.get_value(self.head_addr).ravel()
            # # # 计算饱和度 S = (Head - Bot) / Thickness
            # calc_sat = (current_head - self.botm_arr) / self.cell_thick

            # # # 限制范围 [0, 1]
            # calc_sat = np.clip(calc_sat, 0.0, 1.0)

            # # # 1. 首先处理上限：将承压水或大于1的情况限制为 1.0
            # # calc_sat = np.minimum(calc_sat, 1.0)
            # # # 2. 处理下限阈值：凡是小于 0.1 的（包括负值），一律设为 0.0 使用 np.where(条件, 满足时的值, 不满足时的值)
            # # calc_sat = np.where(calc_sat < 0.01, 0.0, calc_sat)

            # # # import matplotlib.pyplot as plt
            # # # plt.plot(calc_sat[:135])
            # # # plt.show()

            # self.phreeqc_rm.SetSaturation(calc_sat)
            # ! ====== 潜水处理   END ====== #
            
            self.phreeqc_rm.SetConcentrations(conc_from_mf)
            self.phreeqc_rm.SetTime(current_time * 86400.0)
            self.phreeqc_rm.SetTimeStep(dt * 86400.0)
            self.phreeqc_rm.RunCells()

            conc_after_reaction[:] = self.phreeqc_rm.GetConcentrations()

            # ! 矿物输出前置
            temp_selected = self.phreeqc_rm.GetSelectedOutput()
            self.selected_output = temp_selected.reshape(-1, self.nxyz)
            
            # --- Step 4: Write reacted concentrations back to MODFLOW ---
            for i, sp in enumerate(self.components):
                ptr = conc_var_info[sp]["ptr"]
                sl = species_slices[i]
                ptr[:] = conc_after_reaction[sl].reshape(ptr.shape)

            # --- Step 5: Update porosity / read K ---
            if self.if_update_porosity_K:
                old_porosity = self.porosity.copy()
                new_porosity = self._update_porosity()
                self.porosity = new_porosity

                # Update PhreeqcRM porosity
                self.phreeqc_rm.SetPorosity(self.porosity)
                
                # Update GWT porosity / thetam
                for sp in self.components:
                    # self.modflow_api.set_value(self.thetam_tags[sp], new_porosity)
                    self.thetam_ptrs[sp][:] = new_porosity
                                
                current_K11 = self._update_K(current_K11, old_porosity, new_porosity)
                if time_step_index % self.save_interval == 0:
                    self.results_porosity.append(new_porosity)
                    self.results_K.append(current_K11)

            if self.if_update_diffc:
                # Update MODFLOW diffusion coefficient DIFFC
                new_diffc = self._update_diffc(new_porosity)
                if time_step_index % self.save_interval == 0:
                    self.results_diffc.append(new_diffc)
                    
                for sp in self.components:
                    self.modflow_api.set_value(self.diffc_tags[sp], new_diffc)
            
            # --- Save selected output ---
            if time_step_index % self.save_interval == 0:
                self.results.append(self.selected_output.copy())

            time_step_index += 1

            if time_step_index % 20 == 0:
                print(f"  t = {current_time:.2f}/{end_time:.2f} days, step={time_step_index}")

        self.results = np.array(self.results)

        if self.if_update_porosity_K:
            self.results_porosity = np.array(self.results_porosity)
            self.results_K = np.array(self.results_K)

        if self.if_update_diffc:
            self.results_diffc = np.array(self.results_diffc)

        self.final_time_step_index = time_step_index

        print(f"--- Simulation finished, steps={time_step_index}, time={time.time()-start_sim_time:.2f} s ---")

    def run_SIA(self):
        self._initialize_modflow6()
        if not self.is_setup:
            raise RuntimeError("setup() must be called before running the simulation")
        
        print("\n--- Starting reactive transport simulation (SIA with Source Feedback) ---")
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

        if self.if_update_density:
            self.density_addr = self.modflow_api.get_var_address("DENSE", "gwf_model", "BUY")
            self.density_ptr = self.modflow_api.get_value_ptr(self.density_addr)

            prev_density = self.density.copy()
            omega_density = 0.5

        # ======================= IMS for XMI ======================= # 
        n_solutions = self.modflow_api.get_subcomponent_count()
        sol_iters_map = {}
        for solution_id in range(1, n_solutions + 1):
            sln_name = f"SLN_{solution_id}"
            ims_addr = self.modflow_api.get_var_address("MXITER", sln_name)
            sol_iters_map[solution_id] = self.modflow_api.get_value_ptr(ims_addr)
        # ======================= IMS for XMI ======================= # 

        # Preallocate large arrays
        conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float) # ! C_trans
        conc_after_reaction = np.empty_like(conc_from_mf)             # ! C_react
        conc_prev_time = np.zeros_like(conc_from_mf)                  # ! C_trans (上一时刻)

        species_slices = [self._get_species_slice(i) for i in range(self.ncomps)]

        # selected output
        nsel = len(self.phreeqc_rm.GetSelectedOutput()) // self.nxyz

        # Time management
        current_time = self.modflow_api.get_current_time()
        end_time = self.modflow_api.get_end_time()
        time_step_index = 0

        # 获取各 GWT 的孔隙度
        self.thetam_ptrs = {}
        for sp in self.components:
            thetam_addr = self.modflow_api.get_var_address("thetam", f"gwt_{sp}_model", "MST")
            self.thetam_ptrs[sp] = self.modflow_api.get_value_ptr(thetam_addr)

        # ======================= update_porosity_K ======================= #
        if self.if_update_porosity_K:

            self.K11_addr = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
            self.K33_addr = self.modflow_api.get_var_address("K33", "gwf_model", "NPF")

            self.K11_ptr = self.modflow_api.get_value_ptr(self.K11_addr)
            self.K33_ptr = self.modflow_api.get_value_ptr(self.K33_addr)

            current_K11 = self.K11_ptr.copy()

            self.tdis_kper_addr = self.modflow_api.get_var_address("KPER", "TDIS")
            self.tdis_kstp_addr = self.modflow_api.get_var_address("KSTP", "TDIS")

            self.kchangeper_addr  = self.modflow_api.get_var_address("KCHANGEPER", "gwf_model", "NPF")
            self.kchangestp_addr  = self.modflow_api.get_var_address("KCHANGESTP", "gwf_model", "NPF")
            self.nodekchange_addr = self.modflow_api.get_var_address("NODEKCHANGE", "gwf_model", "NPF")
            
            self.modflow_api.set_value(self.nodekchange_addr, np.ones(self.nxyz, dtype=np.int32))

            self.results_porosity = [self.porosity.copy()]
            self.results_K = [current_K11]
        # ======================= update_porosity_K ======================= #

        print(self.modflow_api.get_input_var_names())

        # ! 用于更新扩散系数 ===================== 
        if self.if_update_diffc:
            self.diffc_tags = {
                sp: self.modflow_api.get_var_address("DIFFC", f"gwt_{sp}_model", "DSP")
                for sp in self.components
            }
        # ! 用于更新扩散系数 =====================

        # ======================= SRC ======================= #
        src_var_info = {}
        for sp_name in self.components:
            gwt_model_name = f"gwt_{sp_name}_model"
            src_addr = self.modflow_api.get_var_address("SMASSRATE", gwt_model_name, "SRC")
            src_ptr = self.modflow_api.get_value_ptr(src_addr)
            src_var_info[sp_name] = {
                "ptr": src_ptr
            }
        # ======================= SRC ======================= #

        # 获取体积
        self.area = self.modflow_api.get_value_ptr(self.modflow_api.get_var_address("AREA", gwt_model_name, "DIS"))
        self.top = self.modflow_api.get_value_ptr(self.modflow_api.get_var_address("TOP", gwt_model_name, "DIS"))
        self.botm = self.modflow_api.get_value_ptr(self.modflow_api.get_var_address("BOT", gwt_model_name, "DIS"))
        
        self.volume = self.area * (self.top - self.botm)
        cell_volume = self.volume * self.porosity

        # SIA Parameters
        max_picard_iter = 2000
        # sia_tolerance = 1e-6 # 3e-5
        sia_rtol = 1e-4   # 相对误差容忍度 (0.01%)
        sia_atol = 1e-9  # 绝对误差容忍度 (防止零浓度报错，单位与浓度一致)

        

        while current_time < end_time:

            # ! Step 1 准备阶段
            dt = self.modflow_api.get_time_step()
            self.modflow_api.prepare_time_step(dt)

            if self.if_update_porosity_K and time_step_index > 0:
                # 第 1 步 (index=0) 使用 NPF 文件中的初始值 只有从第 2 步 (index=1) 开始才需要更新 K
                # A. 直接从 TDIS 读取当前正确的 KPER 和 KSTP (1-based) get_value 返回的是数组，直接拿来用即可
                current_tdis_kper = self.modflow_api.get_value(self.tdis_kper_addr)
                current_tdis_kstp = self.modflow_api.get_value(self.tdis_kstp_addr)
                
                # B. 将正确的应力期和步数填入 NPF 的脏标记
                self.modflow_api.set_value(self.kchangeper_addr, current_tdis_kper)
                self.modflow_api.set_value(self.kchangestp_addr, current_tdis_kstp)
                
                self.K11_ptr[:] = current_K11
                self.K33_ptr[:] = current_K11

            self.phreeqc_rm.StateSave(1)
            
            # ! Step 2 保存 t-1 时刻数据 (用于恢复)
            for i, sp in enumerate(self.components):
                ptr = conc_var_info[sp]["ptr"]
                sl = species_slices[i]
                conc_prev_time[sl] = ptr.reshape(-1)
            # print("conc_prev_iter: ", conc_prev_iter.shape)

            # ! 将运移模型初始源汇项设为 0
            # for sp in self.components:
            #     src_var_info[sp]["ptr"][:] = 0.0

            conc_last_picard_iter = np.zeros_like(conc_from_mf) # 存储第 k-1 次迭代结果

            # 在循环外预分配 diff 数组
            diff_buffer = np.empty_like(conc_from_mf)

            # ! Step 3 Picard 迭代
            picard_k = 0
            has_sia_converged = False
            while picard_k < max_picard_iter:
                
                # 0. 迭代前恢复为上一时刻的数据 =======================
                for i, sp in enumerate(self.components):
                    ptr = conc_var_info[sp]["ptr"]
                    sl = species_slices[i]
                    ptr[:] = conc_prev_time[sl].reshape(ptr.shape)

                if self.if_update_density:
                    # current_density = self.phreeqc_rm.GetDensityCalculated() * 1000
                    raw_new_density = self.phreeqc_rm.GetSelectedOutput()[-self.nxyz:]* 1000 # ! 权衡之举

                # A. 运行 MODFLOW6 =======================
                for solution_id in range(1, n_solutions + 1):
                    self.modflow_api.prepare_solve(solution_id)
                    current_max_iter = sol_iters_map[solution_id][0]

                    if (self.if_update_density == True) and (solution_id == 1):
                        # self.density_ptr[:] = current_density
                        relaxed_density = (1.0 - omega) * prev_density + omega * raw_new_density
                        self.density_ptr[:] = relaxed_density
                        prev_density = relaxed_density.copy()


                    kiter = 0
                    has_converged = False
                    while kiter < current_max_iter:
                        has_converged = self.modflow_api.solve(solution_id)
                        kiter += 1
                        if has_converged:                                                                               
                            break
                    self.modflow_api.finalize_solve(solution_id)

                # B. 从 MF6 中获取 C_trans =======================
                for i, sp in enumerate(self.components):
                    ptr = conc_var_info[sp]["ptr"]
                    sl = species_slices[i]
                    conc_from_mf[sl] = ptr #.reshape(-1)

                # conc_from_mf[conc_from_mf < 0.0] = 0.0
                np.maximum(conc_from_mf, 0.0, out=conc_from_mf)

                # C. 进行地球化学反应 =======================
                self.phreeqc_rm.StateApply(1)
                self.phreeqc_rm.SetConcentrations(conc_from_mf)
                self.phreeqc_rm.SetTime(current_time * 86400.0)
                # # print(current_time)
                self.phreeqc_rm.SetTimeStep(dt * 86400.0)
                self.phreeqc_rm.RunCells()

                # D. 获取反应后的浓度 C_react =======================
                conc_after_reaction[:] = self.phreeqc_rm.GetConcentrations()

                # E. 更新源汇项 =======================
                # print(self.components)
                for i, sp in enumerate(self.components):
                    src_ptr = src_var_info[sp]["ptr"]
                    sl = species_slices[i]

                    c_react = conc_after_reaction[sl]
                    c_trans = conc_from_mf[sl]
        
                    # Correction rate
                    if dt > 1e-30:
                        calculated_rate = (c_react - c_trans) * cell_volume / dt
                    else:
                        calculated_rate = 0.0
                    
                    omega = 0.5
                    src_ptr[:] = (1.0 - omega) * src_ptr[:] + omega * calculated_rate
                    # src_ptr[:] = src_ptr[:] + omega * calculated_rate
                
                # print("反应量最大反馈: ", np.max(conc_from_mf-conc_after_reaction))

                # F. 检查是否收敛 =======================
                if picard_k > 0: 
                    # 允许误差阈值 = 绝对误差阈值 + 相对比例 * 当前浓度绝对值
                    # tolerance_threshold = sia_atol + (sia_rtol * np.abs(conc_from_mf))
                    
                    # 避免分配新内存 diff_buffer = conc_from_mf - conc_last_picard_iter
                    np.subtract(conc_from_mf, conc_last_picard_iter, out=diff_buffer)
                    # 逻辑: diff_buffer = abs(diff_buffer)
                    np.abs(diff_buffer, out=diff_buffer)
                    
                    # 计算动态允许误差阈值
                    tolerance_threshold = sia_atol + (sia_rtol * np.abs(conc_from_mf))
                    
                    # 3. 判断是否收敛
                    # 注意：你之前定义的变量是 diff_buffer，不是 diff
                    # if max_diff < tolerance_threshold:
                    if np.all(diff_buffer <= tolerance_threshold):
                        has_sia_converged = True
                        print(f"  -- SIA converged at iter {picard_k}")

                        # for i, sp in enumerate(self.components):
                        #     ptr = conc_var_info[sp]["ptr"]
                        #     sl = species_slices[i]
                        #     ptr[:] = conc_from_mf[sl].reshape(ptr.shape)

                        break
                        
                # conc_last_picard_iter[:] = conc_from_mf[:].copy()
                np.copyto(conc_last_picard_iter, conc_from_mf)
                
                picard_k += 1

            if has_sia_converged == False:
                print(f"  -- SIA not converged")
            
            self.phreeqc_rm.StateDelete(1)

            # --- Step 3: Finalize Time Step ---
            self.modflow_api.finalize_time_step()
            
            # --- Step 4: Save & Log ---
            current_time = self.modflow_api.get_current_time()

            temp_selected = self.phreeqc_rm.GetSelectedOutput()
            self.selected_output = temp_selected.reshape(nsel, self.nxyz)

            if self.if_update_porosity_K:
                old_porosity = self.porosity.copy()
                new_porosity = self._update_porosity()
                self.porosity = new_porosity

                # Update PhreeqcRM porosity
                self.phreeqc_rm.SetPorosity(self.porosity)
                
                # Update GWT porosity / thetam
                for sp in self.components:
                    self.thetam_ptrs[sp][:] = new_porosity
                    
                current_K11 = self._update_K(current_K11, old_porosity, new_porosity)
                
                if time_step_index % self.save_interval == 0:
                    self.results_porosity.append(new_porosity)
                    self.results_K.append(current_K11)

            if self.if_update_diffc:
                # Update MODFLOW diffusion coefficient DIFFC
                new_diffc = self._update_diffc(new_porosity)
                if time_step_index % self.save_interval == 0:
                    self.results_diffc.append(new_diffc)
                    
                for sp in self.components:
                    self.modflow_api.set_value(self.diffc_tags[sp], new_diffc)

            if time_step_index % self.save_interval == 0:
                self.results.append(self.selected_output.copy())

            time_step_index += 1
            if time_step_index % 20 == 0:
                print(f"  t = {current_time:.2f}/{end_time:.2f} days, step={time_step_index}, SIA iters={picard_k}")
            
        self.results = np.array(self.results)
        self.final_time_step_index = time_step_index

        print(f"--- Simulation finished, steps={time_step_index}, time={time.time()-start_sim_time:.2f} s ---")

    def save_results(self, filename: str = None):
        """
        todo Save selected outputs, porosity, and permeability results into .npy files.
        """
        if filename is None:
            # filename = os.path.join(self.output_dir, f"{self.case_name}_results.npy")
            filename = os.path.join(self.output_dir, f"results.npy")
        else:
            filename = os.path.join(self.output_dir, filename)

        base = os.path.splitext(filename)[0]

        if self.headings:
            np.save(filename, np.array(self.results))

            print(f"Results saved to: {filename}")
            
            header_file = base + "_headings.txt"
            with open(header_file, 'w') as f:
                for heading in self.headings:
                    f.write(f"{heading}\n")
            print(f"Headings saved to: {header_file}")
            
            if self.if_update_porosity_K:
                porosity_file = base + "_porosity.npy"
                np.save(porosity_file, self.results_porosity)
                print(f"Porosity results saved to: {porosity_file}")

                k_file = base + "_K.npy"
                np.save(k_file, self.results_K)
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