import os
import sys
import time
import numbers

import numpy as np
# import matplotlib.pyplot as plt
import phreeqcrm
import modflowapi

from pathlib import Path

# 定义一个类型别名, 用于类型提示, 表示可以接受多种类似数组的类型
ArrayLike = int | float | list | tuple | np.ndarray

VM_minerals = { # L/mol
    "Calcite": 0.0369,
    "Dolomite": 0.0645,
    "Halite": 0.0271,
    "Carnallite": 0.1737,
    "Polyhalite": 0.2180,
    "Sylvite": 0.0375,
    "Gypsum": 0.0739,
    "Bischofite": 0.1271,
    "Syngenite": 0.1273,
}

class mf6pqc:
    """
    一个用于耦合 MODFLOW 6 和 PHREEQC-RM 的反应-运移模拟器类。

    这个类封装了从模型设置、初始化、时间步循环、数据交换到
    最终化和结果保存的整个工作流程。
    """
    def __init__(
        self, 
        case_name: str = "Temp_case", 
        nxyz: int = 80, 
        nthreads: int = 3, 
        temperature: ArrayLike = 25.0,
        pressure: ArrayLike = 2.0,
        porosity: ArrayLike = 0.35,
        saturation: ArrayLike = 1.0,
        density: ArrayLike = 1.0,
        print_chemistry_mask: ArrayLike = 1,
        componentH2O: bool = False,
        solution_density_volume: bool = False,
        db_path: str = "input_data/phreeqc.dat", 
        pqi_path: str = "input_data/advect.pqi", 
        modflow_dll_path: str = "C:\\ProgramFiles\\MODFLOW\\libmf6.dll", 
        output_dir: str = "output",
        workspace: str = "./simulation",
        if_update_porosity_K: bool = False,
    ):
        """
        初始化模拟器, 接收定义一个案例所需的所有配置。
        
        Args:
            case_name (str): 案例名称, 用于文件和目录命名。
            nxyz (int): 参与计算的单元格总数。
            nthreads (int): PHREEQC-RM 使用的线程数。
            temperature (ArrayLike): 初始温度 (摄氏度)。可以是标量或长度为 nxyz 的数组。
            pressure (ArrayLike): 初始压力 (atm)。可以是标量或长度为 nxyz 的数组。
            porosity (ArrayLike): 初始孔隙度。可以是标量或长度为 nxyz 的数组。
            saturation (ArrayLike): 初始饱和度。可以是标量或长度为 nxyz 的数组。
            density (ArrayLike): 初始密度 (kg/L)。可以是标量或长度为 nxyz 的数组。
            print_chemistry_mask (ArrayLike): 列表掩码, 决定哪些单元格打印化学信息。
            componentH2O (bool): 是否将 H2O 作为一个独立的传输组分。
            solution_density_volume (bool): 是否使用 PHREEQC 计算的密度。
            db_path (str): PHREEQC 数据库文件路径。
            pqi_path (str): PHREEQC 输入定义文件路径。
            modflow_dll_path (str): libmf6.dll 的路径。
            workspace (str): MODFLOW 模型的工作目录。
        """
        # --- 直接赋予配置 ---
        self.case_name = case_name
        self.nxyz = nxyz
        self.nthreads = nthreads

        self.componentH2O = componentH2O
        self.solution_density_volume = solution_density_volume
        
        # --- 物理属性 (原始值) ---
        self._temperature_in = temperature
        self._pressure_in = pressure
        self._porosity_in = porosity
        self._saturation_in = saturation
        self._density_in = density
        self._print_chemistry_mask_in = print_chemistry_mask

        # --- 文件路径和目录 ---
        self.db_path = db_path
        self.pqi_path = pqi_path
        self.modflow_dll_path = modflow_dll_path
        self.workspace = workspace
        self.output_dir = output_dir

        # --- 实例变量 (将在 setup 阶段创建) ---
        self.phreeqc_rm = None
        self.modflow_api = None
        self.ncomps = None # 参与模拟的化学组分的数量
        self.components = []
        self.headings = []

        # --- 模拟状态 ---

        self.is_setup = False
        self.if_update_porosity_K = if_update_porosity_K
        
        # 存储变量
        self.results = []
        self.results_K = []
        self.results_porosity = []

        # --- 在初始化时处理并验证输入的物理属性 ---
        self._process_physical_properties()
        self._initialize_phreeqcrm()
        print("mf6pqc 实例已创建")
        
    def _set_physical_property(self, name: str, value: ArrayLike) -> np.ndarray:
        """
        通用的函数, 将用户输入的标量或数组转换为符合要求的 ndarray。
        """
        if isinstance(value, numbers.Number):
            return np.full((self.nxyz,), float(value))
        elif isinstance(value, (list, tuple, np.ndarray)):
            arr = np.array(value, dtype=float).ravel()
            if arr.shape != (self.nxyz,):
                raise ValueError(
                    f"提供的'{name}'列表/数组长度为 {arr.shape[0]}, 与网格总数 nxyz ({self.nxyz}) 不匹配。"
                )
            return arr
        else:
            raise TypeError(f"'{name}' 参数类型不支持, 收到了 {type(value).__name__}。")

    def _process_physical_properties(self):
        """将所有物理属性 (标量或序列) 处理成长度为 nxyz 的 np.ndarray"""
        self.temperature = self._set_physical_property("温度", self._temperature_in)
        self.pressure = self._set_physical_property("压力", self._pressure_in)
        self.porosity = self._set_physical_property("孔隙度", self._porosity_in)
        self.saturation = self._set_physical_property("饱和度", self._saturation_in)
        self.density = self._set_physical_property("密度", self._density_in)
        # self.print_chemistry_mask = self._set_physical_property("列表掩码", self._print_chemistry_mask_in).astype(int)
        
    def _initialize_phreeqcrm(self):
        """私有方法：初始化和配置 PhreeqcRM 实例。"""
        print("--- 正在初始化 PHREEQC-RM ---")
        self.phreeqc_rm = phreeqcrm.PhreeqcRM(self.nxyz, self.nthreads)

        os.makedirs(self.output_dir, exist_ok=True)
        prefix = os.path.join(self.output_dir, f"{self.case_name}_prm")
        self.phreeqc_rm.SetFilePrefix(prefix)
        self.phreeqc_rm.OpenFiles()

        # 设置浓度单位
        self.phreeqc_rm.SetUnitsSolution(2)     # ! 溶液单位: 1, mg/L; 2, mol/L; 3, kg/kgs
        self.phreeqc_rm.SetUnitsPPassemblage(0) # ! 矿物相单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsExchange(0)     # ! 离子交换单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsSurface(0)      # ! 表面络合反应单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsGasPhase(0)     # ! 气体单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsSSassemblage(0) # ? 固溶体单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        self.phreeqc_rm.SetUnitsKinetics(0)     # ! 动力学反应物单位: 0, mol/L cell; 1, mol/L water; 2 mol/L rock
        
        # 时间转换 (MODFLOW 使用天, PHREEQC-RM 使用秒)
        self.phreeqc_rm.SetTimeConversion(1.0 / 86400.0)

        # 设置物理属性
        self.phreeqc_rm.SetTemperature(self.temperature)
        self.phreeqc_rm.SetPressure(self.pressure)
        self.phreeqc_rm.SetPorosity(self.porosity)
        self.phreeqc_rm.SetSaturation(self.saturation)
        self.phreeqc_rm.SetDensityUser(self.density) # 注意：这里使用 SetDensityUser

        # 设置其他选项
        self.phreeqc_rm.SetComponentH2O(self.componentH2O)
        self.phreeqc_rm.UseSolutionDensityVolume(self.solution_density_volume)
        # self.phreeqc_rm.SetPrintChemistryMask(self.print_chemistry_mask)

        # 加载数据库和化学定义
        print(f"加载数据库: {self.db_path}")
        self.phreeqc_rm.LoadDatabase(self.db_path)
        
        # 开启化学过程打印输出
        self.phreeqc_rm.SetPrintChemistryOn(True, False, False)
        
        print(f"运行化学定义文件: {self.pqi_path}")
        self.phreeqc_rm.RunFile(True, True, True, self.pqi_path)
        
        # 清理 worker 和 utility 实例的内容
        self.phreeqc_rm.RunString(True, False, True, "DELETE; -all")

        self.ncomps = self.phreeqc_rm.FindComponents()
        
        # 获取参与反应的组分
        self.phreeqc_rm.FindComponents()
        self.components = list(self.phreeqc_rm.GetComponents())
        print(f"参与反应的化学组分列表是: {self.components}")
        
        self.phreeqc_rm.SetScreenOn(False)
        self.phreeqc_rm.SetSelectedOutputOn(True)
        
    def _initialize_modflow6(self):
        """私有方法：初始化 ModflowApi 实例。"""
        print(f"--- 正在初始化 MODFLOW 6 (libmf6.dll) ---")
        print(f"工作目录: {self.workspace}")
        try:
            self.modflow_api = modflowapi.ModflowApi(self.modflow_dll_path, working_directory=self.workspace)
        except Exception as e:
            print(f"错误：无法加载 MODFLOW 6 DLL 或设置工作目录。")
            print(f"DLL路径: {self.modflow_dll_path}")
            print(f"工作目录: {self.workspace}")
            raise e
        self.modflow_api.initialize()

    def _create_ic_array_from_map(self, ic_map: dict) -> np.ndarray:
        """根据 ic_map 字典, 创建并返回一个用于 PHREEQC-RM 的初始条件数组 (ic_array)。"""
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
                print(f"警告：未知的化学模块名称 '{module_name}', 将被忽略。")
                continue
            idx = module_indices[module_name]
            start, end = idx * self.nxyz, (idx + 1) * self.nxyz
            if isinstance(pqi_value, numbers.Number):
                ic_array[start:end] = int(pqi_value)
            elif isinstance(pqi_value, (list, tuple, np.ndarray)):
                arr = np.array(pqi_value, dtype=np.int32).ravel()
                if arr.shape != (self.nxyz,):
                    raise ValueError(f"提供的 '{module_name}' 数组长度与 nxyz 不匹配。")
                ic_array[start:end] = arr
            else:
                raise TypeError(f"'{module_name}' 参数类型不支持。")
        return ic_array

    def _setup_single_ic(self, ic_map: dict):
        """私有方法: 处理单一初始化学条件。"""
        print("--- 正在设置单一初始化学条件 ---")
        ic_array = self._create_ic_array_from_map(ic_map)
        self.phreeqc_rm.InitialPhreeqc2Module(ic_array)

    def _setup_mixed_ic(self, ic_map1: dict, ic_map2: dict, fractions: ArrayLike):
        """私有方法: 处理混合初始化学条件。"""
        print("--- 正在设置混合初始化学条件 ---")
        ic_array1 = self._create_ic_array_from_map(ic_map1)
        ic_array2 = self._create_ic_array_from_map(ic_map2)
        
        # 处理并验证 fractions 输入
        fraction_array = self._set_physical_property("混合比例", fractions)
        # 扩展比例数组以匹配 PhreeqcRM 的要求
        fractions_tiled = np.tile(fraction_array, 7)
        
        self.phreeqc_rm.InitialPhreeqc2Module_mix(ic_array1, ic_array2, fractions_tiled)

    def setup(
        self, 
        ic_map: dict, 
        ic_map2: dict | None = None, 
        fractions: ArrayLike | None = None
    ) -> np.ndarray:
        """
        执行所有必要的设置步骤, 准备开始模拟。

        此方法作为调度器, 根据提供的参数调用相应的私有设置方法。
        - 单一模式: 只提供 `ic_map`。
        - 混合模式: 同时提供 `ic_map`, `ic_map2` 和 `fractions`。

        Args:
            ic_map (dict): 第一个(或唯一的)化学状态映射字典。
            ic_map2 (dict | None, optional): (混合模式) 第二个化学状态映射字典。
            fractions (ArrayLike | None, optional): (混合模式) `ic_map` 状态所占的比例。

        Returns:
            np.ndarray: 计算得到的初始浓度向量。
        """
        if self.is_setup:
            print("警告：模拟器已经设置过。")
            return

        # 1. 通用初始化
        self.is_setup = True

        # 2. 根据参数调度到具体的设置方法
        if ic_map2 is not None and fractions is not None:
            # 调用混合设置
            self._setup_mixed_ic(ic_map, ic_map2, fractions)
        elif ic_map2 is None and fractions is None:
            # 调用单一设置
            self._setup_single_ic(ic_map)
        else:
            raise ValueError("参数不匹配：要使用混合模式, 必须同时提供 'ic_map2' 和 'fractions'。")

        # 3. 通用收尾步骤：运行初始平衡并获取浓度
        print("--- 正在运行初始化学平衡计算 ---")
        self.phreeqc_rm.SetTime(0.0)
        self.phreeqc_rm.SetTimeStep(0.0)
        self.phreeqc_rm.RunCells()
        initial_concentrations = self.phreeqc_rm.GetConcentrations()
        print("--- 初始化学条件设置完成 ---")

        # --- 在这里获取表头，这是最可靠的时机 ---
        self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())
        if not self.headings:
            # 添加一个健壮性检查
            print("警告：未能从 PhreeqcRM 获取到 Selected Output 表头，结果可能无法正确保存。")
    # ---------------------------------------------

        self.selected_output = self.phreeqc_rm.GetSelectedOutput()
        self.selected_output = self.selected_output.reshape(-1, self.nxyz)
        self.results.append(self.selected_output)

        if self.if_update_porosity_K == True:
            self._get_output_information()

        return initial_concentrations
    
    def _get_output_information(self):
        """获取 selected_output 的若干信息"""
        # if not self.headings:
        #     self.headings = list(self.phreeqc_rm.GetSelectedOutputHeadings())

        output_indices = []
        mineral_volumes = []
        d_mineral_names = []

        for idx, heading in enumerate(self.headings):
            # 跳过不以 d_ 开头或长度不够的字段
            if not (heading.startswith("d_") and len(heading) > 2): # > 2 是为了防止 d_ 后面没东西
                continue

            # 提取矿物名称
            d_mineral_name = heading[2:]
            d_mineral_names.append(d_mineral_name)

            output_indices.append(idx)

            # 获取摩尔体积或报错
            if d_mineral_name in VM_minerals:
                mineral_volumes.append(VM_minerals[d_mineral_name])
            else:
                raise ValueError(f"无法找到 '{d_mineral_name}' 的摩尔体积")
            
        self.output_indices = np.array(output_indices, dtype=int)
        self.mineral_volumes = np.array(mineral_volumes, dtype=float).reshape(-1, 1)
        self.d_mineral_names = np.array(d_mineral_names)

    def _update_porosity(self) -> np.ndarray:
        """
        根据矿物相沉淀/溶解的体积变化来计算并返回新的孔隙度。

        Returns:
            np.ndarray: 一个一维数组，包含了每个单元格更新后的孔隙度。
        """
        # self.selected_output 是一个 (n_vars, nxyz) 的数组
        # self.output_indices 包含了所有 d_MineralName 变量的行索引
        mineral_delta_moles = self.selected_output[self.output_indices, :]

        # self.mineral_volumes 是一个 (n_minerals, 1) 的列向量
        # 利用Numpy的广播机制进行逐元素的乘法和求和
        # 计算每个单元格中，所有矿物相变化导致的总体积变化
        total_volume_change = np.sum(self.mineral_volumes * mineral_delta_moles, axis=0)

        # self.porosity 是当前时间步开始时的孔隙度 (一维数组)
        # 计算新孔隙度
        new_porosity = self.porosity * (1.0 - total_volume_change)
        # new_porosity = self.porosity - total_volume_change

        # 修正孔隙度范围，确保其物理意义的有效性 (大于0，小于等于1)
        new_porosity = np.maximum(1e-20, new_porosity)
        new_porosity = np.minimum(1.0,   new_porosity)

        return new_porosity

    def _update_K(self, old_porosity: np.ndarray, new_porosity: np.ndarray):
        """
        使用 Kozeny-Carman 公式更新渗透率。
        此方法直接修改实例变量 self.K。

        Args:
            old_porosity (np.ndarray): 更新前孔隙度的一维数组。
            new_porosity (np.ndarray): 更新后孔隙度的一维数组。
        """
        # self.K 是一个三维数组 (nlays, nrows, ncols)，先将其展平以便计算
        K_flat = self.K.flatten()

        # 为避免除以零导致数值不稳定，给分母加上一个极小值(epsilon)
        # Kozeny-Carman 项: (φ^3) / ((1-φ)^2)
        term_new = new_porosity**3 / ((1 - new_porosity)**2 + 1e-20)
        term_old = (old_porosity**3) / ((1 - old_porosity)**2 + 1e-20)
        
        # 避免除以一个可能是零的 term_old
        # K_new = K_old * (term_new / term_old)
        # 这里的 term_old 不能为零。如果 old_porosity 极小，term_old 会趋于0。
        # 添加一个小的 epsilon 防止除零错误。
        new_K_flat = K_flat * term_new / (term_old + 1e-20)
        
        # 将更新后的渗透系数重塑为原始的三维形状，并更新实例变量
        self.K = new_K_flat.reshape(self.K.shape)

    def _get_species_slice(self, ispecies: int) -> slice:
        """返回第 ispecies 个溶质在一维浓度向量中的切片。"""
        return slice(ispecies * self.nxyz, (ispecies + 1) * self.nxyz)
    # def run(self):
    #     """
    #     执行耦合模拟的主循环。
    #     """
    #     self._initialize_modflow6()
    #     if not self.is_setup:
    #         raise RuntimeError("必须在运行前调用 setup() 方法。")
        
    #     print("\n--- 开始反应-运移耦合模拟 ---")
    #     start_sim_time = time.time()

    #     # --- 优化：在循环外缓存 MODFLOW 变量的地址和形状 ---
    #     print("--- 正在缓存 MODFLOW 变量信息 (地址和形状) ---")
    #     conc_var_info = {}
    #     for sp_name in self.components:
    #         gwt_model_name = f"gwt_{sp_name}_model"
    #         address = self.modflow_api.get_var_address("X", gwt_model_name)
    #         shape = self.modflow_api.get_value(address).shape 
    #         conc_var_info[sp_name] = {"address": address, "shape": shape}
    #         print(f"  - 已缓存溶质 '{sp_name}' 的信息, 形状: {shape}")

    #     if self.if_update_porosity_K:
    #         K_tag = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
    #         # 只在循环开始前读取一次初始 K 值
    #         self.K = self.modflow_api.get_value(K_tag)
    #         print(f"--- 已读取初始渗透系数 K, 形状: {self.K.shape} ---")
            
    #         # 结果列表初始化 (将初始状态的拷贝存入，保证历史记录正确)
    #         self.results_porosity.append(self.porosity.copy())
    #         self.results_K.append(self.K.copy())

    #     current_time = self.modflow_api.get_current_time()
    #     end_time = self.modflow_api.get_end_time()
        
    #     while current_time < end_time:
    #         # --- 第一步：更新 MODFLOW 到下一个时间步 (平流、弥散等) ---
    #         dt = self.modflow_api.get_time_step()
    #         self.modflow_api.update()
    #         current_time = self.modflow_api.get_current_time()
            
    #         # --- 第二步：从 MODFLOW 获取浓度, 交给 PHREEQC-RM ---
    #         conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float)
    #         for isp, sp_name in enumerate(self.components):
    #             var_info = conc_var_info[sp_name]
    #             transported_conc = self.modflow_api.get_value(var_info["address"])
    #             conc_from_mf[self._get_species_slice(isp)] = transported_conc.ravel(order="C")
            
    #         # --- 第三步：在 PHREEQC-RM 中执行化学反应 ---
    #         self.phreeqc_rm.SetConcentrations(conc_from_mf)
    #         self.phreeqc_rm.SetTime(current_time)
    #         self.phreeqc_rm.SetTimeStep(dt)
    #         self.phreeqc_rm.RunCells()
            
    #         # --- 第四步：从 PHREEQC-RM 获取反应后的浓度, 写回 MODFLOW ---
    #         conc_after_reaction = self.phreeqc_rm.GetConcentrations()
    #         for isp, sp_name in enumerate(self.components):
    #             var_info = conc_var_info[sp_name]
    #             updated_conc_slice = conc_after_reaction[self._get_species_slice(isp)]
    #             updated_conc_arr = updated_conc_slice.reshape(var_info["shape"], order="C")
    #             self.modflow_api.set_value(var_info["address"], updated_conc_arr)

    #         # --- 第五步：收集选择性输出结果 ---
    #         # GetSelectedOutput 返回的是新创建的数组，所以这里不需要 .copy()
    #         self.selected_output = self.phreeqc_rm.GetSelectedOutput().reshape(-1, self.nxyz)
    #         self.results.append(self.selected_output)

    #         # --- 第六步：更新孔隙度-渗透系数 ---
    #         if self.if_update_porosity_K:

    #             old_porosity = self.porosity  # 不再复制
    #             new_porosity = self._update_porosity()
    #             self._update_K(old_porosity, new_porosity)

    #             # 赋值并写入模型
    #             self.porosity = new_porosity
    #             self.phreeqc_rm.SetPorosity(self.porosity)
    #             self.modflow_api.set_value(K_tag, self.K)

    #             # 保存当前步结果，需要深拷贝以保持历史
    #             self.results_porosity.append(self.porosity.copy())
    #             self.results_K.append(self.K.copy())

    #         print(f"模拟时间: {current_time:.2f} / {end_time:.2f} 天")
         
    #     end_sim_time = time.time()
    #     print(f"--- 模拟运行完成, 耗时 {end_sim_time - start_sim_time:.2f} 秒 ---")

    def run(self):
        """
        执行耦合模拟的主循环。
        此版本经过优化，使用内存预分配来提高性能。
        """
        self._initialize_modflow6()
        if not self.is_setup:
            raise RuntimeError("必须在运行前调用 setup() 方法。")
        
        print("\n--- 开始反应-运移耦合模拟 ---")
        start_sim_time = time.time()

        # --- 在循环外缓存 MODFLOW 变量的地址和形状 (保持原有优化) ---
        print("--- 正在缓存 MODFLOW 变量信息 (地址和形状) ---")
        conc_var_info = {}
        for sp_name in self.components:
            gwt_model_name = f"gwt_{sp_name}_model"
            address = self.modflow_api.get_var_address("X", gwt_model_name)
            shape = self.modflow_api.get_value(address).shape 
            conc_var_info[sp_name] = {"address": address, "shape": shape}
            print(f"  - 已缓存溶质 '{sp_name}' 的信息, 形状: {shape}")

        # --- 核心优化：预分配结果数组 ---
        # 对于长时程模拟，需要您根据模型的TDIS文件（时间离散化）设置一个保守的
        # 时间步总数的上限。这避免了在循环中动态调整数组大小带来的巨大开销。
        # 示例：假设您的模拟不会超过 5000 个时间步。
        num_timesteps_est = 500
        print(f"--- 预分配内存，预估最大时间步数: {num_timesteps_est} ---")

        # 预分配数组，注意维度中的 +1 是为了存储初始状态 (t=0)
        self.results = np.empty((num_timesteps_est + 1, len(self.headings), self.nxyz), dtype=np.float64)
        # 将 setup() 中计算的初始结果 (t=0) 存入数组的第一个位置
        self.results[0] = self.selected_output 

        if self.if_update_porosity_K:
            K_tag = self.modflow_api.get_var_address("K11", "gwf_model", "NPF")
            self.K = self.modflow_api.get_value(K_tag)
            print(f"--- 已读取初始渗透系数 K, 形状: {self.K.shape} ---")
            
            self.results_porosity = np.empty((num_timesteps_est + 1, self.nxyz), dtype=np.float64)
            self.results_K = np.empty((num_timesteps_est + 1, *self.K.shape), dtype=np.float64)
            # 存储初始状态
            self.results_porosity[0] = self.porosity.copy()
            self.results_K[0] = self.K.copy()
        
        current_time = self.modflow_api.get_current_time()
        end_time = self.modflow_api.get_end_time()
        
        time_step_index = 0
        while current_time < end_time:
            # 安全检查，防止实际步数超出预分配空间
            if time_step_index >= num_timesteps_est:
                print(f"错误：实际时间步数 ({time_step_index}) 已达到预分配上限 ({num_timesteps_est})！")
                print("模拟提前终止。请在 run() 方法中增大 num_timesteps_est 的值。")
                break

            # --- 第一步：更新 MODFLOW 到下一个时间步 ---
            dt = self.modflow_api.get_time_step()
            self.modflow_api.update()
            current_time = self.modflow_api.get_current_time()
            
            # --- 第二步：从 MODFLOW 获取浓度, 交给 PHREEQC-RM ---
            conc_from_mf = np.empty(self.nxyz * self.ncomps, dtype=float)
            for isp, sp_name in enumerate(self.components):
                var_info = conc_var_info[sp_name]
                transported_conc = self.modflow_api.get_value(var_info["address"])
                conc_from_mf[self._get_species_slice(isp)] = transported_conc.ravel(order="C")
            
            # --- 第三步：在 PHREEQC-RM 中执行化学反应 ---
            self.phreeqc_rm.SetConcentrations(conc_from_mf)
            self.phreeqc_rm.SetTime(current_time)
            self.phreeqc_rm.SetTimeStep(dt)
            self.phreeqc_rm.RunCells()
            
            # --- 第四步：从 PHREEQC-RM 获取反应后的浓度, 写回 MODFLOW ---
            conc_after_reaction = self.phreeqc_rm.GetConcentrations()
            for isp, sp_name in enumerate(self.components):
                var_info = conc_var_info[sp_name]
                updated_conc_slice = conc_after_reaction[self._get_species_slice(isp)]
                updated_conc_arr = updated_conc_slice.reshape(var_info["shape"], order="C")
                self.modflow_api.set_value(var_info["address"], updated_conc_arr)

            # --- 第五步：使用预分配数组收集结果 ---
            # PhreeqcRM 返回一个新数组，我们将其整形后存入预分配数组的正确位置
            temp_selected_output = self.phreeqc_rm.GetSelectedOutput()
            # 将当前步结果存入索引 time_step_index + 1 的位置
            self.results[time_step_index + 1] = temp_selected_output.reshape(-1, self.nxyz)
            
            # 必须更新 self.selected_output，因为 _update_porosity 方法依赖它
            self.selected_output = self.results[time_step_index + 1]

            # --- 第六步：更新孔隙度-渗透系数 ---
            if self.if_update_porosity_K:
                old_porosity = self.porosity
                new_porosity = self._update_porosity()
                self._update_K(old_porosity, new_porosity)

                # 赋值并写入模型
                self.porosity = new_porosity
                self.phreeqc_rm.SetPorosity(self.porosity)
                self.modflow_api.set_value(K_tag, self.K)

                # 保存当前步结果到预分配数组
                self.results_porosity[time_step_index + 1] = self.porosity
                self.results_K[time_step_index + 1] = self.K

            time_step_index += 1
            print(f"模拟时间: {current_time:.2f} / {end_time:.2f} 天 (步数: {time_step_index})")
         
        # 记录实际运行的总步数，以便保存结果时使用
        self.final_time_step_index = time_step_index

        end_sim_time = time.time()
        print(f"--- 模拟运行完成, 耗时 {end_sim_time - start_sim_time:.2f} 秒 ---")

    def save_results(self, filename: str = None):
        """
        将收集到的选择性输出结果以及孔隙度和渗透率结果保存到 .npy 文件中。
        此版本经过优化，只保存预分配数组中被实际填充过的部分。
        """
        if not hasattr(self, 'final_time_step_index') or self.final_time_step_index == 0:
            print("警告：没有有效的模拟结果可保存。")
            return

        # 构建文件路径
        if filename is None:
            # filename = os.path.join(self.output_dir, f"{self.case_name}_results.npy")
            filename = os.path.join(self.output_dir, f"results.npy")
        else:
            filename = os.path.join(self.output_dir, filename)
        base = os.path.splitext(filename)[0]

        # 计算实际存储的数据点数量（步数 + 初始状态）
        actual_data_points = self.final_time_step_index + 1

        # 保存选择性输出结果
        if self.headings:
            # 使用切片只保存有数据的部分
            np.save(filename, self.results[:actual_data_points])
            print(f"结果已保存到: {filename} (共 {actual_data_points} 个时间点)")

            # 保存表头
            header_file = base + "_headings.txt"
            with open(header_file, 'w') as f:
                for heading in self.headings:
                    f.write(f"{heading}\n")
            print(f"结果表头已保存到: {header_file}")

            if self.if_update_porosity_K:
                # 保存孔隙度结果
                porosity_file = base + "_porosity.npy"
                np.save(porosity_file, self.results_porosity[:actual_data_points])
                print(f"孔隙度结果已保存到: {porosity_file}")

                # 保存渗透率结果
                k_file = base + "_K.npy"
                np.save(k_file, self.results_K[:actual_data_points])
                print(f"渗透率结果已保存到: {k_file}")
        else:
            print("错误：无法获取结果表头, 无法确定结果维度。")

    def finalize(self):
        """
        完成模拟, 关闭文件并释放资源。
        """
        print("--- 正在结束模拟, 释放资源。 ---")
        if self.modflow_api:
            self.modflow_api.finalize()
            print("MODFLOW API 已关闭。")
        if self.phreeqc_rm:
            self.phreeqc_rm.CloseFiles()
            self.phreeqc_rm.MpiWorkerBreak()
            print("PHREEQC-RM 文件已关闭。")

        self.is_setup = False

    # def save_results(self, filename: str = None):
    #     """
    #     将收集到的选择性输出结果以及孔隙度和渗透率结果保存到 .npy 文件中。

    #     Args:
    #         filename (str, optional): 保存结果的 .npy 文件名。如果为 None,
    #                                   将使用 'case_name_results.npy'。
    #     """
    #     if not self.results:
    #         print("警告：没有结果可保存。")
    #         return

    #     # 构建文件路径
    #     if filename is None:
    #         filename = os.path.join(self.output_dir, f"{self.case_name}_results.npy")
    #     else:
    #         filename = os.path.join(self.output_dir, filename)
    #     base = os.path.splitext(filename)[0]

    #     # 保存选择性输出结果
    #     if self.headings:
    #         num_vars = len(self.headings)
    #         results_array = np.array(self.results).reshape(len(self.results), num_vars, self.nxyz)
    #         np.save(filename, results_array)
    #         print(f"结果已保存到: {filename}")

    #         # 保存表头
    #         header_file = base + "_headings.txt"
    #         with open(header_file, 'w') as f:
    #             for heading in self.headings:
    #                 f.write(f"{heading}\n")
    #         print(f"结果表头已保存到: {header_file}")

    #         if self.if_update_porosity_K == True:
    #             # 保存孔隙度结果
    #             porosity_file = base + "_porosity.npy"
    #             porosity_array = np.array(self.results_porosity)
    #             np.save(porosity_file, porosity_array)
    #             print(f"孔隙度结果已保存到: {porosity_file}")

    #             # 保存渗透率结果
    #             k_file = base + "_K.npy"
    #             k_array = np.array(self.results_K)
    #             np.save(k_file, k_array)
    #             print(f"渗透率结果已保存到: {k_file}")
    #     else:
    #         print("错误：无法获取结果表头, 无法确定结果维度。")

    def get_components(self):
        return list(self.phreeqc_rm.GetComponents())
    
    def get_initial_concentrations(self, number):
        """边界单个获取"""
        bc1 = np.full((1), number)
        #phreeqc_rm.InitialPhreeqc2Concentrations(bc_conc_dbl_vect, bc1)
        return self.phreeqc_rm.InitialPhreeqc2Concentrations(bc1)