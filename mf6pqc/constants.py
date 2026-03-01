SECONDS_PER_DAY = 86400.0  # 一天的秒数，用于时间单位转换
PHREEQCRM_TIME_CONVERSION = 1.0  # PHREEQCRM 模块的时间转换系数（无量纲）
PHREEQCRM_REBALANCE_FRACTION = 0.5  # PHREEQCRM 重新平衡时的权重系数（0–1）
PHREEQCRM_UNITS = {
    "solution": 2,        # 溶液相的单位代码
    "ppassemblage": 0,    # 纯相集合的单位代码
    "exchange": 0,      # 交换相的单位代码
    "surface": 0,       # 表面相的单位代码
    "gas_phase": 0,     # 气相的单位代码
    "ssassemblage": 0,  # 固溶体集合的单位代码
    "kinetics": 0,      # 动力学反应的单位代码
}
MODULE_INDICES = {
    "solution": 0,          # 溶液相在模块列表中的索引
    "equilibrium_phases": 1,  # 平衡相在模块列表中的索引
    "exchange": 2,          # 交换相在模块列表中的索引
    "surface": 3,           # 表面相在模块列表中的索引
    "gas_phase": 4,         # 气相在模块列表中的索引
    "solid_solutions": 5,   # 固溶体在模块列表中的索引
    "kinetics": 6,          # 动力学反应在模块列表中的索引
}
MIN_POROSITY = 1.0e-4  # 允许的最小孔隙度（无量纲）
MAX_POROSITY = 1.0     # 允许的最大孔隙度（无量纲）
MIN_CONCENTRATION = 1.0e-20  # 数值计算中采用的最小浓度阈值（mol/L）
MIN_TIME_STEP = 1.0e-30      # 允许的最小时间步长（秒）
K33_RATIO = 0.1              # 渗透率-孔隙度关系中的 K33 比例系数
SOURCE_RELAXATION = 0.5    # 源项迭代的松弛因子（0–1）
SIA_MAX_PICARD_ITER = 2000  # 序列迭代算法(SIA)的最大 Picard 迭代次数
SIA_RTOL = 1.0e-4           # SIA 的相对收敛容差
SIA_ATOL = 1.0e-9           # SIA 的绝对收敛容差
DENSITY_SCALE = 1000.0      # 密度缩放因子（kg/m³ → 无单位）
DENSITY_RELAXATION = 0.5    # 密度迭代的松弛因子（0–1）
IC_DEFAULT = -1             # 默认初始条件索引（-1 表示未设定）
VM_MINERALS = {
    "Calcite": 0.03693,      # 方解石摩尔体积（L/mol）
    "Dolomite": 0.0645,      # 白云石摩尔体积（L/mol）
    "Halite": 0.0271,        # 石盐摩尔体积（L/mol）
    "Carnallite": 0.1737,    # 光卤石摩尔体积（L/mol）
    "Polyhalite": 0.2180,    # 杂卤石摩尔体积（L/mol）
    "Sylvite": 0.0375,       # 钾盐摩尔体积（L/mol）
    "Gypsum": 0.07421,       # 石膏摩尔体积（L/mol）
    "Bischofite": 0.1271,    # 水氯镁石摩尔体积（L/mol）
    "Syngenite": 0.1273,     # 钙芒硝摩尔体积（L/mol）
    "Ferrihydrite": 0.02399, # 水铁矿摩尔体积（L/mol）
    "Jarosite": 0.15463,     # 黄钾铁矾摩尔体积（L/mol）
    "Gibbsite": 0.03319,     # 三水铝石摩尔体积（L/mol）
    "Siderite": 0.02926,     # 菱铁矿摩尔体积（L/mol）
}
