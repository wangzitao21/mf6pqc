# SIA 与 Strang 架构审查记录

## 审查范围

本次审查覆盖：

- `mf6pqc/coupling/{common,snia,sia,strang,state}.py`；
- 反应后孔隙率、K、扩散和密度反馈的提交时刻；
- MODFLOW 6 静态 TDIS 与 PhreeqcRM 时间/状态 API 的对应关系；
- `Splitting_KineticDecay` 一维解析案例；
- `Splitting_RedoxFront2D` 二维非线性案例。

理论基准为 `references/Steefel1996_Approaches to modeling of reactive transport in porous media.pdf` 中 SNIA、Strang 和 SIA 的方程 (103)–(112)，以及 `references/MIN3P原理手册/MIN3P-THCm.pdf` 中全局隐式耦合、非线性迭代、时间步拒绝/缩短和质量平衡流程。MIN3P 并非 MF6PQC 算法的逐行模板，而是失败策略与长期架构的参照。

## 已发现并修复的问题

| 优先级 | 原问题 | 科学后果 | 修复 |
|---|---|---|---|
| P0 | SIA 每次 Picard 都把已经含旧反应源的 GWT endpoint 再做完整 `dt` 反应 | 同一动力学区间被重复作用；无流一阶衰减得到错误固定点 | 每步保存 `t_n` 完整化学状态；按方程 (110) 移除旧源得到 transport residual base；每次迭代恢复状态后重算 |
| P0 | 部分调用把逻辑步末时间传给 PhreeqcRM `SetTime` | 使用 `TOTAL_TIME` 的 RATES 整体错位一个步长 | `run_reaction_step(start_time, dt)` 统一使用区间起点，SNIA/SIA/Strang/ThermalSNIA 调用同步修改 |
| P0 | SIA 仅靠相邻浓度或松弛后源项变化可能“收敛” | 小松弛因子可制造假收敛，两个 backend endpoint 仍不一致 | 同时检查相邻浓度、transport-reaction 闭合、未松弛源项残差和可选密度残差 |
| P1 | Strang 的反应介质反馈延迟到逻辑步末 | 第二个半 transport step 使用旧 K/孔隙率/扩散/密度，实际不是声明的对称组合 | 中点反应后立即刷新 selected output 和介质属性，强制在第二半步写入 K，并重新计算密度 |
| P1 | Strang 时间表按运行进度才局部发现错误 | 奇数步或不等半步可能在计算开始后才失败 | 运行前一次性验证非空、偶数和相邻等长，并预检逻辑 `save_steps` |
| P1 | SIA 移动水体积忽略饱和度 | 非单位饱和度下源项尺度错误 | 使用 `bulk volume * porosity * configured saturation` |
| P2 | 失败信息只有迭代数 | 无法判断哪个组分阻止闭合 | manifest/异常记录最大浓度、endpoint、源项、密度残差和逐组件闭合 |

## 关键不变量

1. 每个逻辑反应区间只从保存的 `t_n` 动力学状态推进一次；Picard 次数不改变物理反应时间。
2. Strang 的两段 transport TDIS step 必须相等，反应长度为两者之和，反应起点为第一个半步之前的逻辑时间。
3. SIA 只有在 transport endpoint、reaction endpoint 和 reaction source 固定点同时闭合时才提交。
4. 严格模式下任何 MODFLOW solution 或 SIA Picard 失败都中止，不把未收敛场伪装成验证结果。
5. 介质反馈的状态提交点是算法的一部分，不是任意输出操作。

## 定量证据

### 反应-only 不变量

均匀初始 `Spe=1`、`k=100 year^-1`、`dt=1.461 day`（步内
Damköhler 数仍为 0.4）、可忽略输运时：

| 实现 | SIA 平均浓度 |
|---|---:|
| 修复前 | 0.75058745 |
| 修复后 | 0.66900779 |
| SNIA/Strang 同一原生后端 | 0.66900779 |
| 连续解析值 `exp(-0.4)` | 0.67032005 |

修复后的三种算法在原生后端之间一致到 `1e-10`；与连续解析值的小差异属于该 PHREEQC 动力学积分设置，而不是耦合方法重复推进。

### 一维解析分裂案例

案例现已直接采用 Steefel 和 MacQuarrie（1996）方程 (114) 与图 6：
`theta=1`、`v=100 m/year`、`alpha=0.2 m`、`dx=0.4 m`、
`k=100 year^-1`、`t=0.5 year`，并比较 CFL 0.1、0.5、1。参考曲线是
半无限域一级衰减解析解。

按原论文内部节点 `x=j*dx` 约定得到：

| CFL | SNIA RMSE | Strang RMSE | SIA RMSE |
|---:|---:|---:|---:|
| 0.1 | **0.02316** | 0.03016 | 0.02844 |
| 0.5 | **0.01581** | 0.02683 | 0.01935 |
| 1.0 | 0.04982 | 0.02581 | **0.02089** |

模型显式设置 `x=0` 的 CNC 边界节点并只比较其后的 15 个内部节点。SIA 在粗、强分裂
的 CFL=1 下最低，SNIA 的入口过度反应和 Strang 的前沿超前特征也被复现；在较细
CFL 下 SNIA 的时间分裂误差下降，并可与固定空间误差抵消，因而不能把本案例写成
无条件算法排名。MF6PQC 也没有复现原文所称的 SIA 与解析曲线“几乎完全重合”。
MODFLOW 的单元中心有限体积/TVD 与论文控制体有限元不同，输出仍保留 raw DIS
cell-center 误差作为网格坐标敏感性诊断。

### 二维异质含水层案例

`Splitting_RedoxFront2D` 已改为快速、有限固相氧化剂容量控制的二维反应前沿。8 天供体脉冲和 4 天清水冲洗各只使用一个粗逻辑步；基质和斜向反应透镜的 Fe(III) 电子当量容量分别为 `2e-4` 和 `8e-4 model mol/cell`，准一级反应速率为 `5 day^-1`。SNIA 整步越过耗竭前沿，Strang 只在两个半步之间更新一次，而 SIA 在同一时间层内迭代移动供体汇项与固相耗竭状态。

统一指标先分别以入口供体浓度 `1e-3 mol/L` 和最大固相容量 `8e-4 model mol` 归一化，再把两个全场差值拼接计算 RMS；因此排序不是只挑选对 SIA 有利的供体场。

| 方法 | combined NRMSE | Don NRMSE | extent NRMSE | transport solve | reaction evaluation | wall time (s) |
|---|---:|---:|---:|---:|---:|---:|
| SNIA | 0.05947 | 0.07005 | 0.04655 | 2 | 2 | 1.24 |
| Strang | 0.04286 | 0.04469 | 0.04095 | 4 | 2 | 1.31 |
| **SIA** | **0.03637** | **0.03196** | **0.04030** | 104 | 104 | 75.52 |

这使同一综合误差严格满足 `SIA < Strang < SNIA`，而确定性运输工作量严格满足 `SIA > Strang > SNIA`；当前机器上的实测时间也保持相反次序。SIA 两个逻辑步分别用 60 和 44 次 Picard 迭代并严格收敛。`dt=0.25 d` Strang 与 `dt=0.125 d` Strang 的 combined NRMSE 为 0.00150；同样 `dt=0.125 d` 的 SNIA 与细 Strang 差异为 0.00174，用于降低参考方法偏置风险。

该案例是故意设置的强分裂、容量耗竭压力测试，不构成无条件算法排名。二维参考仍是固定空间网格上的细时间步数值解，不是全局隐式或解析真值；平滑动力学和较细耦合步下，Strang 的对称二阶组合仍可能更合适。wall time 受硬件和系统负载影响，发表时应同时报告 `104 > 4 > 2` 的求解次数。

## 仍未解决或仅部分覆盖的风险

1. **瞬态饱和度**：SIA 使用配置 saturation；尚未把 GWF/GWT 的实时 `GWFSAT` 同步给源项水体积和 PhreeqcRM。
2. **正性保持**：强烈生成的移动产物可能使方程 (110) 的代数 intermediate 为负。当前非负域保护加 endpoint 闭合能够拒绝假解，但不是总组分正性保持算法。
3. **时间步回滚**：静态 TDIS 路径没有 MIN3P 式的拒绝—缩步—重启 transaction。
4. **强介质反馈**：SIA 在化学源项收敛后才提交孔隙率/K/扩散；介质—流动—化学尚未进入同一非线性循环。
5. **质量平衡**：现有案例有浓度范数和物理范围，但还应把 GWT budget 与 PhreeqcRM 相转移统一成逐组件守恒账本。
6. **参考解性质**：二维细 Strang 是数值参考，不是全局隐式或解析真值；发表时应同时给出其进一步时间/网格敏感性。

## 长期维护建议

### 下一优先级

1. 建立 `CouplingTransaction`：统一保存/恢复 MODFLOW step、PhreeqcRM state、介质场和输出缓存，为 ATS 与失败重试提供原子边界。
2. 为 SIA 增加可选的总组分/正性保持 formulation；在此之前对明显不可行的负 intermediate 给出专门诊断。
3. 建立逐组件质量账本：入口、出口、GWT storage、SRC、aqueous/solid/kinetic inventories 和数值残差。
4. 将实时 saturation、孔隙率和水体积同步定义成独立 `MobileWaterVolumeProvider`，并用非饱和案例验证。
5. 在保持固定松弛作为基线的同时，评估带 safeguard 的 Aitken/Anderson 加速；任何加速都必须以未加速残差判定收敛。

### 发布门槛

- 50 个以上 backend-free tests 全部通过；
- reaction-only 原生不变量通过；
- 一维默认和时间步扫描通过；
- 二维案例严格 SIA 收敛并输出误差—成本表；
- 至少一个公开复杂化学案例和一个介质反馈案例无回归；
- 记录 MODFLOW、PhreeqcRM、FloPy、Python、线程数、commit 和 dirty-tree 状态。
