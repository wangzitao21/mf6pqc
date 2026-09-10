# MF6PQC v1.0：Zenodo 与 PyPI 发布指南

软件发行版本、CITATION.cff 和 PyPI 使用 `1.0.0`，Git 标签使用 `v1.0.0`；
论文中可以写 MF6PQC v1.0。以下步骤针对同一份审定源码和发行文件。

## 1. 投稿前需要归档什么

GMD 要求投稿时公开可访问的精确版本代码、相关输入和分析脚本，并在
“Code and data availability”中引用持久归档及其 DOI。仅给 GitHub 或 PyPI
地址不足以代替永久归档；论文标题应包括模型名和版本。
参见 [GMD 代码与数据政策](https://www.geoscientific-model-development.net/policies/code_and_data_policy.html)
和[稿件类型](https://www.geoscientific-model-development.net/about/manuscript_types.html)。

软件源码归档应包括 `mf6pqc/`、文档、许可证、打包配置、测试，以及
`examples/` 下的模型、化学数据库、参考数据和绘图脚本。生成的求解器工作目录、
缓存和本地虚拟环境不进入软件包。MODFLOW 共享库另行获取，见
[安装说明](installation.md)。

如果论文使用的软件包之外还有输入、参考结果或图表数据，应另存为论文复现数据归档，
取得单独 DOI 并与软件记录关联。确认每张图可由归档内容重现。
本地 `cases/` 与 `references/` 不属于本次软件发布整理范围。

版本整理没有重新运行数值案例。[历史验证记录](release-readiness.md)保留 E13
pH/Ca 比较未达阈值的情况。投稿时需核对所采用的案例结果、数据来源和软件版本，
不能仅凭 v1.0.0 版本号声称全部案例重新通过验证。

## 2. 延续现有 Zenodo 记录

已有记录是 [MF6PQC v0.1.0](https://zenodo.org/records/18822578)，其版本 DOI
为 `10.5281/zenodo.18822578`。它属于旧版，不能直接填到新版 CITATION.cff 或
用来代表论文使用的 v1.0.0。保留旧记录，在同一系列中增加新版本。

旧记录的文件名和 GitHub Release 链接符合 GitHub 自动归档的形式。
先登录 Zenodo，进入个人菜单 → GitHub，确认 `wangzitao21/mf6pqc` 的连接状态。
从下面两条路线选择一条，不要为同一个版本同时触发两次归档。

### 路线 A：继续使用 GitHub 自动归档

1. 确认仓库已经连接 Zenodo，旧 v0.1.0 记录出现在对应仓库的发布历史中。
2. 审阅并提交清理后的发布文件；保留你自己的未提交研究改动，不要盲目 `git add .`。
   在该提交上创建 `v1.0.0` 标签并推送。
3. 在 GitHub Releases 新建正式 Release，选择 `v1.0.0` 标签，标题可以使用
   “MF6PQC v1.0.0: A Reactive Transport Framework Coupling MODFLOW 6 and PhreeqcRM”。
   写明主要功能、安装方法和已知限制，再发布。
4. Zenodo 自动抓取发布源码。等待完成后打开新记录，确认版本为 `1.0.0`，
   并且 Versions 中仍可访问 v0.1.0。
5. 核对标题、作者、单位、许可证、版本和日期，复制新记录的版本 DOI。
   更新 GitHub 默认分支的引用信息；不要为了回填 DOI 移动已发布标签。

Zenodo 的 GitHub 路线不支持预先保留 DOI。不要把手动草稿预留的 DOI 混入这条路线。
官方说明：[连接仓库](https://help.zenodo.org/docs/github/enable-repository/)、
[归档 Release](https://help.zenodo.org/docs/github/archive-software/github-upload/)、
[GitHub DOI 预留限制](https://support.zenodo.org/help/en-gb/24-github-integration/73-can-i-pre-reserved-a-doi-before-a-github-release)。

### 路线 B：在旧记录上手工增加版本

1. 在旧记录页面点击 **New version**，进入新版草稿，不要通过 Edit 把旧版本改名。
2. 如希望 DOI 已经写入最终代码包，使用草稿的 **Get a DOI now!** 预留新版 DOI，
   填入 CITATION.cff 的 `doi`，并在实际发布时填写 `date-released`。
   随后重新检查、构建，保证上传的是包含最终元数据的源码。
3. 上传审定的源码 ZIP，或包含源码、文档和案例的 `.tar.gz`；wheel 可作为附加文件。
   如果草稿导入了旧文件，删除草稿内旧 v0.1.0 ZIP，保留真正的新版本内容。
4. 填写下表中的元数据并预览。确认文件内容和版本后点击 **Publish**。
5. 若仓库连接了 GitHub 自动归档，应先暂停该仓库的自动归档，再建立对应 GitHub
   Release，避免重复生成 Zenodo 记录。以后继续选择一种一致的归档路线。

官方说明：[管理版本](https://help.zenodo.org/docs/deposit/manage-versions/)、
[创建上传与预留 DOI](https://help.zenodo.org/docs/deposit/create-new-upload/)。

| 字段 | 内容 |
|---|---|
| Resource type | Software |
| Title | MF6PQC v1.0.0: A Reactive Transport Framework Coupling MODFLOW 6 and PhreeqcRM |
| Version | 1.0.0 |
| Creators | Zitao Wang；按实际贡献补充其他软件作者，并核对单位与 ORCID |
| License | GNU General Public License v3.0 only，与仓库 GPL-3.0-only 一致 |
| Access | Open；投稿时文件须公开可访问 |
| Publication date | 实际发布日期 |
| Related works | 精确 GitHub 标签链接、论文复现数据 DOI；论文取得 DOI 后补入论文链接 |

Description 可说明：MF6PQC couples MODFLOW 6 flow and solute transport with
PhreeqcRM chemistry, with SNIA, SIA and Strang coupling and optional porosity,
hydraulic conductivity, diffusion and density feedback. The archive includes
source code, documentation and example inputs. Python requirements and native
solver setup are described in the installation guide.

论文应引用 **v1.0.0 的版本 DOI**，不是 v0.1.0 DOI；覆盖所有版本的 Concept DOI
适合项目总入口。不要凭记录编号猜 Concept DOI，从页面 Versions 区域复制。
参见 [Zenodo DOI 版本规则](https://zenodo.org/help/versioning)。
审稿期间若代码或数据变化，为修改版增加新版本并更新论文引用；论文发表后可编辑
软件记录的相关文献元数据，增加论文 DOI，无须因这一元数据更新另发软件版本。

## 3. 准备 PyPI 账号

包发布到 **PyPI**，用户通过 **pip** 安装。[PyPI](https://pypi.org/account/register/)
和 [TestPyPI](https://test.pypi.org/account/register/) 分别注册，验证邮箱并启用
双因素认证。两站的账号与 API token 独立。

确认 `mf6pqc` 名称能由你的账号使用；页面不存在也不保证名称一定可注册。
首次手动上传通常用 Account settings → API tokens 创建 Entire account token；
项目建立后可改为仅限该项目的 token。token 在终端隐藏提示中输入，不写入仓库。
参见 [PyPI API token](https://pypi.org/help/#apitoken)。

## 4. 构建与本地安装检查

在干净的源码目录打开 PowerShell（仓库根目录含 `pyproject.toml`），使用 Python 3.11+：

```powershell
python -m venv .venv
$releasePy = Join-Path $PWD '.venv/Scripts/python.exe'
& $releasePy -m pip install --upgrade pip
& $releasePy -m pip install '.[dev,examples]'
& $releasePy -m unittest discover -s tests -v
& $releasePy -m ruff check mf6pqc tests examples scripts
& $releasePy -m ruff format --check mf6pqc tests examples scripts
& $releasePy scripts/check_examples.py
$version = '1.0.0'
$buildStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$distDir = Join-Path $PWD "dist/$version-$buildStamp"
& $releasePy -m build --outdir $distDir
$wheel = (Get-ChildItem -LiteralPath $distDir -Filter '*.whl').FullName
$sdist = (Get-ChildItem -LiteralPath $distDir -Filter '*.tar.gz').FullName
& $releasePy -m twine check --strict $wheel $sdist
& $releasePy scripts/check_distribution.py $distDir
```

这些命令不运行原生数值案例。每步成功后再继续。
如果已拿到本次检查生成的 `dist/1.0.0`，且此后源码没有修改，可将 `$distDir`
设为那个目录，直接使用已检查的 wheel 和 sdist，无须重复构建。

建立独立环境，从本地 wheel 安装；正常运行依赖从正式 PyPI 解析：

```powershell
python -m venv .venv-verify
$verifyPy = Join-Path $PWD '.venv-verify/Scripts/python.exe'
& $verifyPy -m pip install --index-url https://pypi.org/simple/ $wheel
& $verifyPy -m pip check
& $verifyPy -I -c "import mf6pqc; print(mf6pqc.__version__); print(mf6pqc.__file__)"
```

应打印 `1.0.0`，路径应位于 `.venv-verify` 的 site-packages。
`-I` 避免误从当前源码目录导入；该检查不初始化求解器。

## 5. 上传 TestPyPI

```powershell
& $releasePy -m twine upload --repository testpypi --username __token__ $wheel $sdist
```

提示时输入 **TestPyPI token**。确认页面上的版本、说明和两个文件正确，然后测试下载：

```powershell
& $verifyPy -m pip uninstall -y mf6pqc
& $verifyPy -m pip install --index-url https://test.pypi.org/simple/ --no-deps --no-cache-dir "mf6pqc==$version"
& $verifyPy -m pip check
& $verifyPy -I -c "import mf6pqc; print(mf6pqc.__version__); print(mf6pqc.__file__)"
```

这里的 `--no-deps` 依赖上一步已从正式 PyPI 安装运行依赖，避免从测试站解析第三方包。
TestPyPI 不会自动同步到正式 PyPI，也不适合永久归档。
参见 [TestPyPI 用法](https://packaging.python.org/en/latest/guides/using-testpypi/)。

## 6. 正式上传 PyPI

确认 TestPyPI 安装通过，用 **正式 PyPI token** 上传同一份 wheel 和 sdist：

```powershell
& $releasePy -m twine upload --repository pypi --username __token__ $wheel $sdist
```

只上传 `.whl` 和 `.tar.gz`，不要将源码 ZIP 或校验和文件传给 Twine。
成功后用户可安装：

```powershell
python -m pip install mf6pqc==1.0.0
```

发布后再从正式 PyPI 的全新环境检查下载、`pip check` 和隔离导入。
PyPI 已经使用过的文件名不能通过删除重用；代码变动后用 `1.0.1` 等新版本重新构建。
官方教程：[打包与上传](https://packaging.python.org/en/latest/tutorials/packaging-projects/)、
[文件名不能重用](https://pypi.org/help/#file-name-reuse)。

`pip install` 不安装 MODFLOW 共享库，也不把 `examples/` 放入 site-packages；
运行案例需要下载源码归档并按[安装说明](installation.md)配置原生求解器。

## 7. 可选：GitHub Trusted Publishing

已有 `.github/workflows/release.yml` 可在 Actions 中手动执行并指定 `v1.0.0` 标签。
先在 PyPI 配置 publisher：owner `wangzitao21`，repository `mf6pqc`，workflow
`release.yml`，environment `pypi`；并创建对应 GitHub environment。项目尚未建立时
可以使用 pending publisher。此路线使用 OIDC，无须保存 PyPI token。
参见 [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)。

手动 Twine 与发布工作流二选一，同一版本不要重复上传。
