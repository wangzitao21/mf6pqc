<p align="center">
  <img src=".github/assets/mf6pqc-logo.png" width="430" alt="mf6pqc：穿过含水层网格的水流与矿物反应" />
</p>

<h1 align="center">MF6PQC</h1>
<p align="center"><b>连接地下水流动、溶质运移与水–岩反应</b></p>
<p align="center">基于 MODFLOW 6 与 PhreeqcRM 的 Python 反应运移框架</p>

<p align="center">
  <a href="https://pypi.org/project/mf6pqc/"><img src="https://img.shields.io/pypi/v/mf6pqc?color=087e8b" alt="PyPI" /></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-3.11%2B-164b68" alt="Python 3.11+" /></a>
  <a href="https://github.com/MODFLOW-ORG/modflow6/releases/tag/6.8.0"><img src="https://img.shields.io/badge/MODFLOW-6.8.0-087e8b" alt="MODFLOW 6.8.0" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-GPL--3.0-d69b39" alt="GPL 3.0" /></a>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#案例与论文图">案例与论文图</a> ·
  <a href="#复现绘图">复现绘图</a> ·
  <a href="#扩展自己的模型">扩展自己的模型</a>
</p>

MF6PQC 在保留 FloPy 和 PHREEQC 建模方式的基础上，协调水流、运移、化学反应及物性更新。它面向需要比较水文地球化学反馈的研究：溶液密度如何改变流动，矿物溶解与沉淀如何改变孔隙度，以及这些变化如何进一步影响渗透系数和扩散。

<table>
  <tr>
    <th>非均质含水层中的反应运移</th>
    <th>蒸发驱动的密度环流</th>
    <th>优势通道与整体响应</th>
  </tr>
  <tr>
    <td><a href="examples/PHT3D_E10/plot.ipynb"><img src="examples/PHT3D_E10/output/figures/figure_05.png" width="270" alt="二维含水层反应运移结果" /></a></td>
    <td><a href="examples/Hamann2015/plot.ipynb"><img src="examples/Hamann2015/output/figures/figure_06.png" width="270" alt="盐湖密度分布与地下水流线" /></a></td>
    <td><a href="examples/Article_Channel2D/plot.ipynb"><img src="examples/Article_Channel2D/output/figures/figure_12.png" width="270" alt="卤水开采中的优势通道和反馈响应" /></a></td>
  </tr>
</table>

## 模型如何连接

```mermaid
flowchart LR
    A["FloPy<br/>网格、边界与运移参数"] --> B["MODFLOW 6<br/>GWF · GWT"]
    C["PHREEQC 输入<br/>溶液、矿物与反应动力学"] --> D["PhreeqcRM<br/>逐单元化学反应"]
    B -->|运移后的组分浓度| D
    D -->|反应后的组分浓度| B
    D --> E["密度与介质物性更新"]
    E -->|密度 · 孔隙度 · 渗透系数 · 扩散系数| B
    F["MF6PQC<br/>时间推进、耦合策略与结果管理"] -.-> B
    F -.-> D
```

| 能力 | 内容 |
| --- | --- |
| 耦合算法 | SNIA、SIA 与 Strang 分裂；可比较精度、迭代次数和计算开销 |
| 化学过程 | 使用 PhreeqcRM 管理 PHREEQC 溶液、平衡相、动力学及其他受支持的反应实体 |
| 密度反馈 | 将化学计算得到的溶液密度传入 MODFLOW 6 的密度耦合路径 |
| 介质演化 | 根据矿物体积变化更新孔隙度，并按配置更新渗透系数和扩散系数 |
| 温度与黏度 | 可选 GWE/VSC 路径；热耦合案例展示温度、反应速率和黏度的相互影响 |
| 结果管理 | 保存组分清单、时间、反应结果及启用的物性；新运行附带环境与输入文件摘要 |

各案例独立设置反馈项。开启某种反馈需要相应的物理假设、参数及 MODFLOW 包配置。

## 快速开始

### 安装 Python 环境

需要 Python 3.11 或更新版本。使用发布的库：

```bash
python -m pip install mf6pqc
```

复现仓库案例时，获取完整仓库并安装案例依赖：

```bash
git clone https://github.com/wangzitao21/mf6pqc.git
cd mf6pqc
python -m pip install -e ".[examples]"
```

### 配置 MODFLOW 6.8.0

从 [USGS 官方 MODFLOW 6.8.0 发布页](https://github.com/MODFLOW-ORG/modflow6/releases/tag/6.8.0) 下载与操作系统相符的发行包，将其中的动态库和可执行文件放到仓库的 `bin/mf6.8.0/`。PyPI 安装不会自动安装 MODFLOW 动态库。

| 操作系统 | 动态库 | 可执行文件 |
| --- | --- | --- |
| Windows | `libmf6.dll` | `mf6.exe` |
| Linux | `libmf6.so` | `mf6` |
| macOS | `libmf6.dylib` | `mf6` |

也可以通过 `MF6PQC_BIN` 指定二进制所在目录，或分别设置 `MF6PQC_LIBMF6` 与 `MF6PQC_MF6_EXE`。例如在 PowerShell 中：

```powershell
$env:MF6PQC_BIN = 'C:\modflow\mf6.8.0\bin'
python examples/PHT3D_E01/run.py
```

所有案例默认使用 6.8.0。Windows 官方发行包中的程序会显示构建字符串 `6.8.0+8680167.dirty`；其 `libmf6.dll` 的 SHA-256 为 `624995b7f592dacd52abe37e1c10c5ff52ecde1d9b188447343c17396e69bda2`。核对版本时应同时检查发行包来源和文件摘要。

### 运行第一个案例

在仓库根目录执行：

```bash
python examples/PHT3D_E01/run.py
jupyter lab examples/PHT3D_E01/plot.ipynb
```

`run.py` 构建并运行耦合模型，`plot.ipynb` 读取保存结果、绘图并进行数值核对。也可以进入案例目录后执行 `python run.py`。

## 案例与论文图

`examples/` 包含 22 个独立案例。每个案例的顶层结构一致：

```text
案例名/
├── modflow_model.py   # 网格、边界、物理参数及 MODFLOW 模型
├── run.py            # 化学初始化、耦合设置与运行入口
├── plot.ipynb        # 结果读取、可视化与数值核对
├── input_data/       # PHREEQC 输入、数据库、初始场与参考数据
├── output/           # 反应与物性结果、无损归档和导出的图
└── simulation/       # MODFLOW 输入、数值输出及无损归档
```

| 案例 | 主要内容 | 论文图 |
| --- | --- | --- |
| [Splitting_KineticDecay](examples/Splitting_KineticDecay/plot.ipynb) | 动力学衰减问题中的 SNIA、SIA、Strang 精度与开销 | 图 3 |
| [PHT3D_E03](examples/PHT3D_E03/plot.ipynb) | 含铁碳酸盐体系的一维反应运移 | 图 4 |
| [PHT3D_E10](examples/PHT3D_E10/plot.ipynb) | 非均质含水层中的二维多组分反应运移 | 图 5 |
| [Hamann2015](examples/Hamann2015/plot.ipynb) | 蒸发浓缩、密度环流与蒸发岩矿物分带 | 图 6–7 |
| [Xie2015_B3](examples/Xie2015_B3/plot.ipynb) | 多矿物反应、孔隙度及水力响应 | 图 8 |
| [Xie2015_B4](examples/Xie2015_B4/plot.ipynb) | 矿物分布与有效扩散系数反馈 | 图 9 |
| [Article_Channel2D](examples/Article_Channel2D/plot.ipynb) | 非均质蒸发岩溶浸中的四种反馈组合 | 图 10–12 |
| [PHT3D_E01](examples/PHT3D_E01/plot.ipynb)–[PHT3D_E13](examples/PHT3D_E13/plot.ipynb) | PHT3D 基准系列；各案例保留独立输入和参考数据 | 含上述图 4–5 |
| [Xie2015_B1](examples/Xie2015_B1/plot.ipynb)、[Xie2015_B2](examples/Xie2015_B2/plot.ipynb) | 矿物反应与水力性质演化的补充对照 | — |
| [Splitting_RedoxFront2D](examples/Splitting_RedoxFront2D/plot.ipynb) | 二维氧化还原前沿的分裂误差比较 | — |
| [GWE_VSC_Reactive](examples/GWE_VSC_Reactive/plot.ipynb) | 温度、反应与黏度反馈 | — |

图 1–2 为概念与程序结构示意图，不对应独立的案例运行结果。

## 复现绘图

每本 `plot.ipynb` 都可以从上到下独立执行。绘图代码只读取本案例内的数据，不依赖外部论文目录。论文图保留原有配色、字体、面板布局、单位与输出尺寸，导出到 `output/figures/`。

仓库在 `output/` 和 `simulation/` 中提供 `saved_results*.zip` 无损归档。笔记本首先补齐缺失的归档文件，然后读取原始数组、水头或流量预算；已存在的结果不会被归档覆盖。因此，仅重绘保存结果无需启动 MODFLOW 或 PHREEQC，也无需配置 MODFLOW 动态库。

重新计算时运行案例的 `run.py`，随后重新执行 `plot.ipynb`，即可绘制新结果。Hamann、Xie、PHT3D_E11–E13 和卤水案例计算量较大，可先使用归档结果检查绘图和数据结构。历史保存结果保留其原始数值，不应据此声称已用新的求解器版本重新计算。

结果数组的解释以案例内字段为准：`results_headings.txt` 给出变量顺序，`results_times.npy` 或案例的时间配置给出保存时刻。矿物的体积基准、孔隙水浓度和密度单位在绘图代码中显式转换。PHT3D 与 MIN3P 参考结果来自原案例的对照数据；精度偏差也在笔记本中保留显示。

### 卤水案例的四种情景

| 标签 | 密度反馈 | 孔隙度–渗透系数反馈 |
| --- | :---: | :---: |
| S00 | 关闭 | 关闭 |
| S10 | 开启 | 关闭 |
| S01 | 关闭 | 开启 |
| S11 | 开启 | 开启 |

```bash
# 依次执行四种情景；相同配置且已完成的标签会直接复用。
python examples/Article_Channel2D/run.py

# 单独创建一个新标签，保留原情景结果。
python examples/Article_Channel2D/run.py --scenario S11 --label S11_repeat --threads 2

# 从现有四种情景的结果重绘论文图。
jupyter lab examples/Article_Channel2D/plot.ipynb
```

该案例采用固定体积、准稳态水力调整与经验孔隙度–渗透系数关系。其边界、化学体系、质量核对和适用范围在笔记本中说明。

## 扩展自己的模型

从物理过程接近的案例开始，复制完整案例目录，并依次修改：

1. 在 `modflow_model.py` 中设置网格、边界、水力参数、时间离散和运移包。
2. 在 `input_data/` 中配置化学数据库、PHREEQC 输入、初始场与对照数据。
3. 在 `run.py` 中设置化学分区、组分映射、耦合算法和物性反馈。
4. 在 `plot.ipynb` 中核对时间、单位、守恒关系及参考误差，再解释模型结果。

网格单元数、浓度排列、化学分区及保存时刻需要彼此一致。比较算法或反馈机制时，应明确保持哪些输入和边界相同。

## 检查与依赖

```bash
# 检查全部案例的目录、笔记本语法与无求解器副作用的导入。
python scripts/check_examples.py

# 对选定案例进行独立回归计算，结果写入指定目录。
python scripts/check_examples.py --native PHT3D_E01 PHT3D_E08 --output-root .release-checks/native
```

回归命令提供 PHT3D_E01–E10、GWE 及两个分裂算法案例。长时案例由各自的 `run.py` 显式启动。完整依赖范围见 [pyproject.toml](pyproject.toml)；一个已验证的 Windows 组合为 Python 3.12.9、NumPy 2.2.1、FloPy 3.10.0、modflowapi 1.0.1、PhreeqcRM 0.0.18 和 MODFLOW 6.8.0。

## 许可与引用

MF6PQC 采用 [GPL-3.0-only](LICENSE) 许可。MODFLOW 6、PhreeqcRM、化学数据库与基准研究分别保留其作者归属和适用条款。研究中使用本项目时，请注明所用版本，并引用相应求解器及具体案例的原始文献。

- [MF6PQC 源代码与版本](https://github.com/wangzitao21/mf6pqc)
- [MODFLOW 6.8.0 软件引用](https://doi.org/10.5066/P1PGE9XW)
- [问题反馈](https://github.com/wangzitao21/mf6pqc/issues)

图标以含水层网格、水流路径与矿物晶体表现 MF6PQC 的耦合对象。
