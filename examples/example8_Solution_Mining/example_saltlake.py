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
    ic_mapping = {
        'solution':           0,
        'equilibrium_phases': 1,
        'exchange':           1,
    }
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
    field_root = "./input_data/mycase/init_hk"

    # 2. 找到所有子目录下的 .npy 文件
    pattern = os.path.join(field_root, "*", "*.npy")
    npy_paths = glob.glob(pattern)
    print(f"[*] 搜索模式：{pattern}")
    print(f"[*] 一共找到 {len(npy_paths)} 个 .npy 文件：")
    for p in npy_paths:
        print("    ", p)
    if not npy_paths:
        print("[!] 没有找到任何 .npy 文件，请检查路径是否正确。")
        return

    # 3. 循环处理每个随机场
    for npy_path in npy_paths:
        fname = os.path.basename(npy_path) # e.g. "gaussian_181.npy"
        case_name = os.path.splitext(fname)[0] # "gaussian_181"
        print(f"\n==== 开始 Case: {case_name} ====")

        try:
            # 更新 sim_params
            sim_params = base_sim_params.copy()
            sim_params["case_name"] = case_name
            sim_ws = os.path.join("./simulation", case_name)
            sim_params["workspace"] = sim_ws
            os.makedirs(sim_ws, exist_ok=True)

            # 读取并 reshape K 网格
            print(f"[*] 载入随机场文件：{npy_path}")
            K_flat = np.load(npy_path)
            K_grid = K_flat.reshape(100, 400, 1)

            # 创建并初始化模拟器
            print("[*] 初始化 mf6pqc 模型...")
            simulator = mf6pqc(**sim_params)
            initial_conc = simulator.setup(ic_map=ic_mapping)
            bc_conc = simulator.get_initial_concentrations(1)
            components = simulator.get_components()

            # 获取密度
            rho = simulator.phreeqc_rm.GetDensityCalculated() * 1000  # kg/m³
            print(f"[*] 初始密度：{rho} kg/m³")

            # 运行 transport model
            print("[*] 构建并运行 transport_model...")
            transport_model(
                sim_ws=sim_ws,
                species_list=components,
                initial_conc=initial_conc,
                bc=bc_conc,
                porosity=sim_params["porosity"],
                K11=K_grid,
                initial_density=rho,
                initial_head=100.0
            )

            # 运行并保存
            print("[*] 运行模拟...")
            simulator.run()
            print("[*] 保存结果...")
            simulator.save_results()
            simulator.finalize()

            print(f"==== 完成 Case: {case_name} ====")

        except Exception as e:
            print(f"[ERROR] Case {case_name} 运行出错：")
            traceback.print_exc()

if __name__ == "__main__":
    print("[*] 脚本启动，当前工作目录：", os.getcwd())
    main()
