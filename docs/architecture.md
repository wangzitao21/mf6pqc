# MF6PQC 架构与耦合算法设计

## 1. 架构目标

MF6PQC 的首要约束是科研可追溯性，而不是隐藏数值细节。长期维护应遵守四条原则：

1. 已验证案例的数值路径、单位和输出时间必须可追踪；
2. MODFLOW 6 与 PhreeqcRM 的状态所有权、时间语义和数组排列必须显式；
3. 耦合算法、介质反馈和原生后端生命周期相互解耦；
4. 每项科学主张都应有后端无关单元测试和至少一个原生集成案例支撑。

当前实现采用稳定的 `MF6PQC` facade、独立耦合策略和共享后端原语。案例负责建立与 PhreeqcRM 组件一一对应的 GWT 模型；算法层负责推进时间、交换状态和记录诊断。

## 2. 模块职责

| 模块 | 职责 | 不应承担 |
|---|---|---|
| `mf6pqc/mf6pqc.py` | 用户入口、配置兼容、生命周期、结果清单 | 某一算法的具体循环 |
| `mf6pqc/backends.py` | 创建和释放 PhreeqcRM/MODFLOW API | 耦合顺序 |
| `mf6pqc/coupling/common.py` | TDIS 展开、浓度交换、反应推进、求解器和保存原语 | 算法专有迭代 |
| `mf6pqc/coupling/snia.py` | `T(dt) -> R(dt)` | SIA 源项迭代 |
| `mf6pqc/coupling/strang.py` | `T(dt/2) -> R(dt) -> T(dt/2)` 与配对预检 | SIA 收敛控制 |
| `mf6pqc/coupling/sia.py` | 源项 Picard 状态、残差和严格失败策略 | SNIA/Strang 顺序 |
| `mf6pqc/coupling/state.py` | 算法运行态字段所有权 | 科学公式 |
| `mf6pqc/feedback.py` | 孔隙率、K、扩散、密度和边界电导反馈 | 化学反应求解 |
| `mf6pqc/input_processing.py` | 化学初始条件索引的校验与打包 | 时间推进 |
| `mf6pqc/output_processing.py` | selected output 解释与可靠保存 | 耦合算法 |

组件浓度始终采用 PhreeqcRM 的 component-major 排列：先存一个组件的所有单元，再存下一个组件。每个 GWT 模型拥有一个组件的空间场。

## 3. 时间和状态所有权

PhreeqcRM 的 `SetTime` 表示反应区间起点，`SetTimeStep` 表示积分长度；PhreeqcRM 不替 MF6PQC 推进全局时间。所有算法必须调用：

```text
SetTime(t_n)
SetTimeStep(dt)
RunCells()
```

而不能把 `t_n + dt` 误传为反应起点。这个约束影响在 `RATES` 中使用 `TOTAL_TIME` 的动力学模型。

状态所有权如下：

- GWT 的 `X` 指针拥有当前被输运的组件浓度；
- PhreeqcRM 拥有溶液物种、动力学反应物、平衡相、交换/表面等化学状态；
- SIA 在每个逻辑步开始保存一次 PhreeqcRM 状态，每次 Picard 迭代先恢复该状态；
- MODFLOW TDIS 拥有可执行的静态时间表，MF6PQC 不在初始化后伪造额外 transport step。

## 4. 三种耦合策略

### SNIA

```text
C^n --T(dt)--> C^T --R(dt)--> C^(n+1)
```

反应从 `t_n` 开始推进 `dt`。介质属性在反应后更新，影响下一逻辑步。该路径是兼容默认值，优点是稳健和成本低；缺点是刚进入边界的质量也承受完整反应时间，粗步长下可能过度反应。

### Strang

```text
C^n --T(dt/2)--> C* --R(dt)--> C** --T(dt/2)--> C^(n+1)
```

静态 TDIS 无法在 API 初始化后自由拆步，因此每两个相邻、相等的 TDIS step 被解释为一个逻辑 Strang step。运行前必须一次性验证：步数非零且为偶数、每对半步相等、`save_steps` 不越界。

反应产生的孔隙率、K、扩散或密度属于中点状态，必须在第二个 transport 半步前写入；把反馈延迟到逻辑步末会破坏上述算子组合。第二半步后，MF6PQC 以零反应时间重新配位以生成 endpoint selected output，同时回滚该诊断操作，避免再次推进动力学状态。

### SIA

SIA 使用 GWT 的保留名为 `SRC` 的源项包实现 Steefel 与 MacQuarrie 方程 (108)–(112) 的源项 Picard 形式。第 `m` 次迭代执行：

```text
1. 恢复 C^n，使用 q_R^m 求解隐式 transport endpoint C^(m+1)
2. 构造 transport residual base:
       C_base = C^(m+1) - q_R^m * dt / V_water
3. 恢复逻辑步起点的 PhreeqcRM 完整状态
4. 从 C_base 在 [t_n, t_n + dt] 重新计算反应 endpoint
5. q_candidate = (C_reacted - C_base) * V_water / dt
6. q_R^(m+1) = q_R^m + omega * (q_candidate - q_R^m)
```

`V_water = bulk_volume * porosity * saturation`。每次化学计算都从保存的 `t_n` 状态开始，因此 Picard 迭代不会把同一个动力学区间重复推进多次。

收敛必须同时满足：

1. 相邻 transported iterate 的浓度变化；
2. `C_reacted - C_transported` 的 endpoint 闭合；
3. 未乘松弛系数的源项残差；
4. 启用密度反馈时的密度残差。

仅检查“松弛后的更新量”会在小松弛系数下制造假收敛。用于发表的案例应启用 `sia_fail_on_nonconvergence=True`，保存每步迭代次数和残差。

## 5. SIA 的适用边界

源项 SIA 是对全局隐式耦合的模块化近似，不等同于组装一个整体 Jacobian。需要明确以下边界：

- `C_base` 是方程 (110) 的代数 transport residual，不一定是可直接观测的物理中间态；强烈生成的移动产物可能使它为负；
- 当前实现会按组件定义执行非负域保护，但 endpoint 闭合仍必须通过，否则严格模式会失败；
- 若出现持续负中间态或 Picard 不收敛，应减小 TDIS 步长，或改用总组分/全局隐式的正性保持形式，不能仅增加最大迭代数；
- 介质属性在 SIA 收敛后提交，不在源项 Picard 内迭代；因此“反应—孔隙率—流场”强反馈尚不是完全耦合 Newton 解；
- 当前移动水体积使用配置的 saturation，尚未把瞬态 `FMI/GWFSAT` 指针纳入已验证契约。

## 6. 失败、重启与自适应时间步

MIN3P 的全局隐式流程可以在非线性失败后缩短并重启时间步。MF6PQC 当前从静态 TDIS 展开时间表，MODFLOW API 初始化后不能任意改写该表。因此：

- `fail_on_nonconvergence` 控制 MODFLOW solution 的严格失败；
- `sia_fail_on_nonconvergence` 独立控制 SIA Picard 的严格失败；
- 非严格 SIA 会提交最后一个反应 endpoint，但在 `sia_diagnostics` 和 manifest 中标记退化；
- 发表计算应预先做时间步敏感性分析并使用严格模式；
- 真正的“拒绝、缩步、回滚、重算”需要单独设计 ATS/transaction 契约，不能用循环外修改 `dt` 冒充。

## 7. 验证矩阵

| 风险 | 最小验证资产 |
|---|---|
| 反应时间起点错误 | mock 调用顺序测试 + 使用 `TOTAL_TIME` 的化学测试 |
| SIA 重复推进动力学状态 | `Splitting_KineticDecay/reaction_only_check.py` |
| SNIA/Strang/SIA 算子顺序 | `tests/test_splitting_algorithms.py` |
| SIA 假收敛 | 源项、endpoint 闭合和未松弛残差单元测试 |
| Strang 半步表错误 | TDIS 配对预检测试 |
| 一维分裂误差 | `Splitting_KineticDecay` 解析解与时间步扫描 |
| 二维非线性和成本 | `Splitting_RedoxFront2D` 细步长参考与工作量统计 |
| 介质反馈顺序 | Strang 中点 K/密度调用顺序测试及 GWE/VSC 案例 |

图形检查只能用于发现问题，不能作为通过条件。每个发布案例必须输出数值范数、物理范围、迭代统计、后端版本和参考来源。

## 8. 扩展新算法的规则

1. 在 `mf6pqc/coupling/` 新建独立策略模块；
2. 复用 `common.py` 的后端原语，不直接复制生命周期代码；
3. 用 dataclass 明确拥有全部运行态；
4. 写出算子顺序、时间起点、状态保存/恢复和介质反馈提交点；
5. 在 `CouplingMethod` 注册并保持现有入口兼容；
6. 同时增加快速 mock 测试、原生不变量测试和一个量化案例；
7. 若需要 ATS、回滚或全局 Jacobian，先定义后端 transaction 边界，再实现算法。
