import os
import numpy as np
import traceback
import phreeqcrm
import modflowapi
import sys

# 确保能导入 mf6pqc 和 transport_model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from mf6pqc.mf6pqc import mf6pqc
from modflow_models import transport_model

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
        "db_path": "./examples/example9_jool/input_data/pitzer.dat",
        "pqi_path": "./examples/example9_jool/input_data/input1.pqi",
        "modflow_dll_path": r"C:\ProgramFiles\MODFLOW\libmf6.dll",
        "if_update_porosity_K": True,
    }

    npy_path = "./examples/example9_jool/input_data/gaussian.npy"

    case_name = os.path.splitext(os.path.basename(npy_path))[0]
    sim_ws = os.path.join("./examples/example9_jool/simulation")
    os.makedirs(sim_ws, exist_ok=True)
    
    print(f"\n==== 运行 Case: {case_name} ====")
    try:
        # 更新 sim_params
        sim_params = base_sim_params.copy()
        sim_params.update({
            "case_name": case_name,
            "workspace": sim_ws,
        })

        # 读取并 reshape K 网格
        print(f"[*] 加载随机场：{npy_path}")
        K_grid = np.load(npy_path).reshape(100, 400, 1)

        # 初始化并设置模型
        simulator = mf6pqc(**sim_params)
        initial_conc = simulator.setup(ic_map=ic_mapping)
        bc_conc = simulator.get_initial_concentrations(1)
        components = simulator.get_components()

        # 获取初始密度（kg/m³）
        rho = simulator.phreeqc_rm.GetDensityCalculated() * 1000
        # print(f"[*] 初始密度：{rho:.2f} kg/m³")

        # 构建并运行 transport_model
        print("[*] 构建 transport_model 并运行")
        transport_model(
            sim_ws=sim_ws,
            species_list=components,
            initial_conc=initial_conc,
            bc=bc_conc,
            porosity=sim_params["porosity"],
            K11=1, #K_grid,
            initial_density=rho,
            initial_head=100.0
        )

        # 运行并保存结果
        print("[*] 运行 PHREEQC-RTF 模拟")
        simulator.run()
        print("[*] 保存结果")
        simulator.save_results()
        simulator.finalize()

        print(f"==== Case {case_name} 完成 ====")

    except Exception:
        print(f"[ERROR] Case {case_name} 运行出错：")
        traceback.print_exc()

if __name__ == "__main__":
    print("[*] 脚本启动，工作目录：", os.getcwd())
    main()
