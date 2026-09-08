# 发布前检查记录 — 2026-09-07

工程整理、短程回归和本地发行包验证已完成。**当前仍有一个科学验收阻断项：PHT3D_E13 的 pH/Ca 与现有参考数组不满足原有阈值。** 因此版本保留为 `0.2.0.dev0`，未发布到 PyPI，不能将本轮工作表述为全部案例已重新验证通过。

## 已完成的工程改进

- 版本号单一来源、现代许可证元数据、明确的包内容、类型标记、源码包和 wheel 构建检查。
- 补齐安装、API、贡献、发布流程、变更记录和引用元数据；新增 CI 与手动触发的发布工作流。
- 22 个案例统一为 `input_data/`、`output/`、`simulation/`，以及 `modflow_model.py`、`run.py`、`plot.ipynb` 和可选的 `validate.py`、`plot.py`。
- 所有案例驱动可安全导入，使用稳定的案例路径；新增独立运行目录和跨平台库文件名选择。既有科学输入和参考数组未修改。
- SaltLake 配置归并到 `modflow_model.py`；分析/对照归并到 `validate.py`；绘图/通道诊断归并到 `plot.py`。其功能保留在案例层。
- 关闭案例默认逐单元化学打印，保留数值输出；库内使用标准日志，命令行案例显式启用进度日志。

## 已修复的问题

1. 整数参数、保存/反应步号和初始条件不能静默截断，检查 int32 上下界、非有限值和二值掩膜。
2. PhreeqcRM 负返回码统一转为异常；错误或中断触发清理；文件关闭失败时仍尝试结束工作线程；已结束实例不可重用。
3. 接近 1 的 TDIS 几何倍率保留其真实几何步长；拒绝不支持的非 DAYS 时间单位及 ATS。
4. 计算密度独立读取，不再覆盖无关 selected-output 列；selected-output 数量变化立即报错。
5. 结果保存前验证物理范围和 JSON 元数据；逐文件原子写入，manifest 最后写入；补齐软件版本、输入和原生库 SHA-256、时间轴含义。
6. 保留历史类名、主要构造参数、耦合方法和导入接口，没有向核心层加入单个案例的拟合参数或条件分支。

## 检查结果

| 检查 | 结果及范围 |
|---|---|
| 单元回归 | 73 项通过；修改前为 58 项；不加载原生求解器 |
| Ruff lint / format | 通过 |
| E01 绘图 notebook | 在独立目录完整执行通过；源 notebook 保持无执行输出 |
| 全案例静态检查 | 22 个案例的目录、notebook 语法及受保护导入通过；不等于数值验证 |
| MODFLOW 构造参数审计 | 22 个模型的科学构造参数保留；变化为路径和等价字符串格式 |
| PHT3D_E01 | 官方参考校验通过；浓度结果和时间轴与修改前逐元素一致 |
| GWE_VSC_Reactive | 温度同步、VSC、反应、孔隙度和 K 检查通过 |
| PHT3D_E04 | 交换案例校验通过；四组分 NRMSE 约 1.04%–1.81% |
| PHT3D_E08 | 四组分官方参考校验通过 |
| Splitting_KineticDecay | 仅反应一致性通过；三方法最大差 2.02×10⁻¹²，对解析解最大误差 8.51×10⁻⁷ |
| SaltLake_Brine3D | 252 单元、2 年 smoke feedback 通过；流量误差和反馈恒等式通过 |
| wheel / sdist | 构建、Twine 严格元数据检查、内容检查通过 |
| 独立 wheel 安装 | 从 site-packages 导入；版本一致；导入本身不加载原生库；E01/GWE 安装版验证通过 |

本机验证环境为 Windows、Python 3.12.9、NumPy 2.2.1、modflowapi 0.2.0、PhreeqcRM 0.0.18、FloPy 3.10.0；按案例使用 MODFLOW 6.7.0 或 6.8.0。CI 的其他平台/版本矩阵尚未在远端执行。本轮后端无关测试的分支覆盖统计约为 58%，原生集成检查独立执行，不计入该覆盖数字。

## E13 发布阻断

| 指标 | 当前 NRMSE | 原验收上限 |
|---|---:|---:|
| pH | 0.067094 | 0.040 |
| Ca | 0.090854 | 0.045 |

在独立目录用修改前核心代码重跑 E13，得到与修改后**全部结果逐元素一致**的输出；原有偏差可以重现。因此它不是本次重构引入的数值变化，但仍不能把 E13 标为通过。本轮未改动其化学参数、参考数组或验收阈值。

下一步需要核对参考数组对应的 PHT3D 原始输入、数据库、输出时刻及模型约定。已向作者请求可使用的原始输入位置。也查询了 [PHT3D 官方下载页](https://www.pht3d.org/pht3d_exe.html)，其公开包包含 13 个案例，但本机下载请求返回 HTTP 406，未取得该包。原始输入核实后再决定是否需要科学模型修正。

E13 的 `validate.py` 会记录全部指标到 `output/validation.json`，并以非零状态退出，避免只显示第一个失败量或把失败隐藏为通过。

## 严格未运行的范围

Xie2015_B1、B2、B3、B4、PHT3D_E11 和 Hamann2015 未执行模拟。本轮也未重跑完整的 Splitting_KineticDecay / Splitting_RedoxFront2D 研究及其他未选短案例；对它们只做静态审查。`article`、`cases`、`references` 未用于此次修改或科学核对。

这些未运行案例保留原有科学参数，但在完成单独的数值验收前，不能声称其结果已被本轮重新验证。

## 本地证据和复现

机器可读概要见 [release-verification.json](release-verification.json)。详细临时日志、基线数组和独立运行结果已于清理时移至被 Git 忽略的 `trash/.release-checks/`（后续重跑会重新生成 `.release-checks/`）。它们不进入发行包。

```bash
python -m unittest discover -s tests -v
python -m ruff check mf6pqc tests examples scripts
python -m ruff format --check mf6pqc tests examples scripts
python scripts/check_examples.py
python scripts/check_examples.py --native PHT3D_E01 GWE_VSC_Reactive
python scripts/check_examples.py --native PHT3D_E13
python -m build
python -m twine check --strict dist/*
python scripts/check_distribution.py dist
```

E13 命令当前应失败，这是保留的真实验收结果。稳定发布前需解除该阻断，并由作者按需求单独组织长案例验收。
