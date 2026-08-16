# MF6PQC 科学与单位契约

本文档描述软件所假定的科学含义。代码通过测试保证实现一致性，但模型适用性仍需案例作者论证。

## 1. 量与单位

| 量 | MF6PQC/后端契约 |
|---|---|
| MODFLOW 时间 | 由 TDIS 决定；当前 examples 使用 day |
| PhreeqcRM 时间 | MF6PQC 将 TDIS 时间乘以 86400 后以 second 传入 |
| 浓度 | 数值必须与 GWT 和 PhreeqcRM 的案例定义一致；现有案例通常使用 mol/L 的数值 |
| 密度输入 | 公共 `density` 字段为 kg/L |
| BUY 密度 | 写入 MODFLOW 时转换为 kg/m³ |
| NPF `K11/K33` | 水力传导系数，单位为 MODFLOW length/time |
| `d0`/`DIFFC` | MODFLOW length²/time；若模型为 m、day，则应为 m²/day |
| 矿物摩尔体积 | L/mol |
| 孔隙率、饱和度 | 无量纲 |

不得仅因变量名相同就混用 SI 单位与模型单位。案例 README 应列出 length、time、mass/concentration 三套单位。

## 2. 浓度排列

PhreeqcRM 浓度向量采用 component-major：

```text
[component_0(cell_0 ... cell_n), component_1(cell_0 ... cell_n), ...]
```

MF6PQC 在初始化时按 PhreeqcRM 返回的 `components` 顺序缓存每个 GWT 模型。组件映射到 MODFLOW 模型名后若发生大小写不敏感的名称碰撞，运行将停止，而不是静默串线。

## 3. 孔隙率反馈

对 selected output 中每个 `d_<mineral>`：

```text
delta_volume_fraction = delta_moles * molar_volume
phi_new = clip(phi_old - sum(delta_volume_fraction), 1e-4, 1.0)
```

正的矿物增量表示固相体积增加，因此孔隙率减小。该式成立的前提是 PhreeqcRM 固相量的基准体积与摩尔体积相乘后确实得到代表性总体积的比例。默认摩尔体积只是案例默认值；若数据库相定义或文献值不同，必须通过 `mineral_molar_volumes` 覆盖并记录来源。

`porosity_update_mask=False` 的单元仍可参与化学反应，但保持介质孔隙率与由其驱动的 K 不变，适合某些规定边界控制体。其科学合理性需由案例说明。

## 4. 水力传导系数反馈

默认 Kozeny–Carman 比值为：

```text
K_new = K_old * [phi_new³/(1-phi_new)²] / [phi_old³/(1-phi_old)²]
```

这里的 `K` 是 MODFLOW NPF 水力传导系数，不是内禀渗透率。密度由 BUY 处理，未来黏度应由 VSC 处理。`FluidAdjustedKozenyCarmanUpdater` 是显式 opt-in 模型；若同时使用 BUY/VSC，可能重复计算流体性质影响，不能作为默认选择。

默认 `K33 = k33_ratio * K11` 是历史行为，不是普适地层规律。三维各向异性案例应明确配置或扩展独立张量更新。

## 5. 扩散反馈

当前关系为：

```text
DIFFC = phi^(1/3) * d0
```

这是一个特定的孔隙率—曲折度经验关系，不代表所有介质。未来应像 K 更新器一样策略化；在此之前，案例必须说明为何采用该指数。

## 6. 密度反馈

优先使用 PhreeqcRM `GetDensityCalculated()`，并将 kg/L 乘以 1000 写入 BUY。兼容模式可以读取 selected output 的最后一行，但必须通过 `density_output_heading` 明确验证其名称，避免因为 USER_PUNCH 顺序变化把其他量当作密度。

## 7. 耦合时间语义

- 对所有方法，PhreeqcRM `SetTime` 接收反应区间起点 `t_n`，`SetTimeStep` 接收积分长度；PhreeqcRM 本身不推进全局时间；
- SNIA：先完成一个 MODFLOW transport step，再从上一次反应时刻开始、以累计的 `dt` 运行反应并写回；
- SIA：在同一准备好的 MODFLOW step 内，用反应修正源项做 Picard 迭代；每次化学迭代先恢复逻辑步起点状态，禁止重复推进同一动力学区间；
- Strang：相邻两个相等的 TDIS step 分别作为前后半步，中间反应推进两者之和；反应产生的介质反馈在第二个半步前提交。

SIA 的 GWT 约定要求每个组件模型存在名为 `SRC` 的源项包。移动水体积为 `bulk cell volume * porosity * configured saturation`。瞬态非饱和模型中的实时 `FMI/GWFSAT` 尚未纳入已验证同步路径，不能把配置饱和度解释成已经自动跟踪 MODFLOW 饱和度。

当前实现从静态 TDIS `PERLEN/NSTP/TSMULT` 展开时间表。尚未为 ATS 定义“被拒绝/缩短的 transport step 如何与反应时间一致”的契约，因此不能把 ATS 当作已验证功能。

## 8. 收敛和失败

MODFLOW 每个 solution 的 `MXITER` 被尊重。失败会写入 `modflow_convergence_failures`；`fail_on_nonconvergence=True` 时立即抛出 `ConvergenceError`。

SIA 同时检查：

1. 相邻 Picard 迭代的 transported concentration 差；
2. reaction endpoint 与 transported endpoint 的闭合差；
3. 未乘松弛系数的 reaction source-rate 残差；
4. 启用密度反馈时的密度残差。

容差由 `sia_atol`、`sia_rtol` 和移动水体积共同形成。不得用“松弛后的源项更新很小”代替原始固定点残差。SIA 未收敛时可记录并继续，也可通过 `sia_fail_on_nonconvergence=True` 严格终止；失败记录包含逐组件 endpoint 闭合，便于识别某个生成/消耗组分导致的不可行中间态。用于发表的结果建议采用严格模式，并报告每步迭代统计及最大残差。

当前静态 TDIS 路径不能像 MIN3P 那样在非线性失败后自动拒绝、缩短并重启时间步。严格失败后的推荐处理是修改输入 TDIS 并完整重跑；在定义事务式回滚契约前，不应把运行中临时改变 `dt` 声称为已验证 ATS。

## 9. 每个新案例最低检查

- 水量与关键组分质量平衡；
- 网格和时间步收敛性；
- 初始/边界化学定义索引；
- selected output 行与单位；
- 反应前后非负性及电荷误差解释；
- 孔隙率、K、密度和扩散的物理范围；
- 与解析解、原软件输出或公开观测的量化误差；
- 所用数据库、MODFLOW、PhreeqcRM 和 MF6PQC 版本。
