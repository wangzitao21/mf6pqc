import os
import glob
import numpy as np
import traceback
import phreeqcrm
import modflowapi
from scr.mf6pqc import mf6pqc
from modflow_models.saltlake_example import transport_model

def main():
    # 初始条件映射 & 公共模拟参数模板
    ic_mapping_1 = {
        'solution':           0,
        'equilibrium_phases': 1,
        'exchange':           1,
    }
    ic_mapping_2 = {
        'solution':           0,
        'equilibrium_phases': 2,
        'exchange':           1,
    }

    fractions = np.load("./input_data/mycase/init_hk/mix/minerials_negative.npy").ravel()

    base_sim_params = {
        "nxyz": 100*400*1,
        "nthreads": 6,
        "temperature": 25.0,
        "pressure": 2.0,
        "porosity": 0.30,
        "saturation": 1.0,
        "density": 1.277848,
        "print_chemistry_mask": 1,
        "componentH2O": False,
        "solution_density_volume": True,
        "db_path": "./input_data/mycase/pitzer.dat",
        "pqi_path": "./input_data/mycase/input1.pqi",
        "modflow_dll_path": r"C:\ProgramFiles\MODFLOW\libmf6.dll",
        "if_update_porosity_K": True,
    }

    # 1. 定义所有随机场所在的根目录
    # field_root = "./input_data/mycase/init_hk"

    K_flat = np.load("./input_data/mycase/init_hk/mix/K.npy")
    K_grid = K_flat.reshape(100, 400, 1)

    case_name = "mix_negative"
    base_sim_params["case_name"] = case_name
    sim_ws = os.path.join("./simulation", case_name)
    base_sim_params["workspace"] = sim_ws

    simulator = mf6pqc(**base_sim_params)
    initial_conc = simulator.setup(ic_map=ic_mapping_1, ic_map2=ic_mapping_2, fractions=fractions)
    bc_conc = simulator.get_initial_concentrations(1)

    components = simulator.get_components()

    rho = simulator.phreeqc_rm.GetDensityCalculated() * 1000  # kg/m³
    print(f"[*] 初始密度：{rho} kg/m³")

    transport_model(
        sim_ws=sim_ws,
        species_list=components,
        initial_conc=initial_conc,
        bc=bc_conc,
        porosity=base_sim_params["porosity"],
        K11=K_grid,
        initial_density=rho,
        initial_head=100.0
    )

    simulator.run()
    simulator.save_results()
    simulator.finalize()

    print("\n-------------------------------------------")
    print(f"模拟 '{base_sim_params['case_name']}' 执行完毕。")
    print("-------------------------------------------\n")

if __name__ == "__main__":
    print("[*] 脚本启动，当前工作目录：", os.getcwd())
    main()
