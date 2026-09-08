# 第一次发布 MF6PQC：Windows PowerShell 操作指南

以下命令由你手动执行；每一段成功后再执行下一段，有报错就先停止处理。上传命令会向外部发布，本文没有代你执行。

包是发布到 **PyPI** 上，用户再通过 **pip** 安装。TestPyPI 是独立的测试站点，不会自动同步到正式 PyPI。

当前版本为 `0.2.0.dev0`。E13 的 pH/Ca 仍未通过既有参考阈值，见 [发布检查报告](release-readiness.md)。现在可以练习 TestPyPI 流程；正式稳定发布前必须核对并解决科学验收问题，记录未重跑的长案例范围。不要仅删除版本号的 `dev0` 就当作科学验收已完成。

## 1. 注册两个账号

- [正式 PyPI 注册](https://pypi.org/account/register/)
- [TestPyPI 注册](https://test.pypi.org/account/register/)

分别注册、验证邮箱，按站点要求启用双因素认证并保存恢复码。两站账号和 token 相互独立。

检查项目名 `mf6pqc` 是否能由你的账号使用。未找到项目页面也不能保证名称一定可注册，以站点上传校验为准。若名称被他人占用，需要改 `pyproject.toml` 中的发布名，并同步调整本文的安装和文件名；Python 导入名可以仍为 `mf6pqc`。

手动首次上传使用 API token：进入对应网站 Account settings → API tokens → Add API token。项目尚不存在时选择 Entire account；首次上传建立项目后，创建仅限 `mf6pqc` 的 token 并撤销临时全账号 token。保存完整的 `pypi-...`，不要提交到 Git，也不用发送给我。详情见 [PyPI 认证说明](https://pypi.org/help/#apitoken)。

## 2. 准备一个干净环境

打开 PowerShell，执行：

```powershell
Set-Location 'D:\Archives\Python\mf6pqc'
python --version
python -m venv .venv
$releasePy = Join-Path $PWD '.venv\Scripts\python.exe'
& $releasePy -m pip install --upgrade pip
& $releasePy -m pip install -e '.[dev,examples]'
```

使用 Python 3.12（本机已验证）；项目声明最低为 3.11。上轮检查环境已经移入 `trash`，请新建环境，不要直接运行移动后的虚拟环境。命令直接指定解释器，不需要执行 Activate.ps1，也不需要调整 PowerShell 执行策略。

## 3. 确认版本和项目资料

检查以下文件：

- `mf6pqc/_version.py`：唯一版本号来源。演练可以保留 `0.2.0.dev0`；科学验收完成后的首次正式版可用 `0.2.0`。
- `pyproject.toml`：作者、邮箱、仓库地址、Python 要求、依赖及许可证信息是否符合你的实际意图。
- `README.md`、`CHANGELOG.md`：公开介绍、安装方法和这一版改动。
- `docs/release-readiness.md`：真实验证范围和未解决问题。

PyPI 上已经使用过的上传文件名不能通过删除重用。若上传后修改代码，增加版本号并重新构建；例如开发版用 `0.2.0.dev1`，已发布正式版的修复用 `0.2.1`。不要覆盖同版本旧文件。参见 [PyPI 文件名规则](https://pypi.org/help/#file-name-reuse)。

## 4. 运行检查并构建

```powershell
& $releasePy -m unittest discover -s tests -v
& $releasePy -m ruff check mf6pqc tests examples scripts
& $releasePy -m ruff format --check mf6pqc tests examples scripts
& $releasePy scripts/check_examples.py
```

以上最后一条只做静态检查，不运行慢案例。需要复核安装环境的短程数值行为时执行：

```powershell
& $releasePy scripts/check_examples.py --native PHT3D_E01 GWE_VSC_Reactive
```

E13 尚未解除的失败不能被这两项短检查替代。这里不提供运行 Xie B1–B4、E11、Hamann2015 的命令。

读取当前版本，为本次构建创建唯一目录，避免混入旧包：

```powershell
$version = (& $releasePy -c "from mf6pqc._version import __version__; print(__version__)").Trim()
$buildStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$distDir = Join-Path $PWD "dist\$version-$buildStamp"
& $releasePy -m build --outdir $distDir
& $releasePy -m twine check --strict "$distDir/*"
& $releasePy scripts/check_distribution.py $distDir
Get-ChildItem -LiteralPath $distDir
```

应得到一个 `.whl` 和一个 `.tar.gz`。wheel 用于安装核心包，源码包还包括文档、测试和案例输入；`trash`、原生库、模拟生成结果不进入发行包。工具流程依据 [PyPA 打包教程](https://packaging.python.org/en/latest/tutorials/packaging-projects/)。

先验证本地 wheel 确实可安装：

```powershell
python -m venv .venv-verify
$verifyPy = Join-Path $PWD '.venv-verify\Scripts\python.exe'
& $verifyPy -m pip install --upgrade pip
$wheel = (Get-ChildItem -LiteralPath $distDir -Filter '*.whl').FullName
& $verifyPy -m pip install $wheel
& $verifyPy -m pip check
& $verifyPy -I -c "import mf6pqc; print(mf6pqc.__version__); print(mf6pqc.__file__)"
```

最后打印的路径应指向 `.venv-verify/Lib/site-packages/mf6pqc`。`-I` 避免误把当前目录源码当成安装版。导入通过只证明安装有效，数值正确性仍由案例校验保证。

## 5. 上传 TestPyPI 并测试下载

以下命令使用 **TestPyPI token**：

```powershell
& $releasePy -m twine upload --repository testpypi --username __token__ "$distDir/*"
```

提示输入 API token/密码时粘贴完整 token；输入不显示字符属正常情况。不要输入网页登录密码。成功后打开 [TestPyPI 的 mf6pqc 页面](https://test.pypi.org/project/mf6pqc/)，检查说明、版本和两个下载文件。

卸载刚才的本地安装，再从测试站安装这一版本：

```powershell
& $verifyPy -m pip uninstall -y mf6pqc
& $verifyPy -m pip install --index-url https://test.pypi.org/simple/ --no-deps --no-cache-dir "mf6pqc==$version"
& $verifyPy -m pip check
& $verifyPy -I -c "import mf6pqc; print(mf6pqc.__version__); print(mf6pqc.__file__)"
```

前一步安装本地 wheel 已从正常 PyPI 安装运行依赖，因此这里使用 `--no-deps`，只从 TestPyPI 下载 MF6PQC，避免测试站缺少依赖导致误判。TestPyPI 的数据也可能被清理，不用它长期分发正式包。参见 [TestPyPI 官方说明](https://packaging.python.org/en/latest/guides/using-testpypi/)。

## 6. 正式发布到 PyPI

仅在科学验收和上述安装检查完成后执行。

如果刚才演练的是 `0.2.0.dev0`，先将 `_version.py` 改为正式版号并更新变更记录，然后重新执行第 4–5 步，用**同一份通过 TestPyPI 检查的正式版构建文件**上传 PyPI。不要把旧开发包当成正式包。

以下命令改用 **正式 PyPI token**：

```powershell
& $releasePy -m twine upload --repository pypi --username __token__ "$distDir/*"
```

成功后打开 [PyPI 的 mf6pqc 页面](https://pypi.org/project/mf6pqc/)。首次成功上传会创建该项目，不需要提前手动建立项目页面。

再次验证正式站下载：

```powershell
& $verifyPy -m pip uninstall -y mf6pqc
& $verifyPy -m pip install --index-url https://pypi.org/simple/ --no-cache-dir "mf6pqc==$version"
& $verifyPy -m pip check
& $verifyPy -I -c "import mf6pqc; print(mf6pqc.__version__); print(mf6pqc.__file__)"
```

正式版本发布后，用户即可执行 `python -m pip install mf6pqc`。

## 7. 保存版本与告诉用户如何真正运行

把发布所对应的源码和变更记录提交到 Git，在该提交上创建匹配版本的 tag，再推送并建立 GitHub Release。若工作区含其他研究文件，提交前逐项检查，不要盲目 `git add .`。GitHub Release 可附上该版验收范围和安装说明。

`pip install mf6pqc` 会安装 Python 运行依赖，但不会替用户安装 MODFLOW 6 共享库，也不会把 `examples` 安装进 site-packages。需告诉用户另行准备匹配的 `libmf6.dll/.so/.dylib`、化学数据库和模型输入；案例从源码仓库或源码包获取。具体步骤见 [安装及原生后端说明](installation.md)。

仓库已有 `.github/workflows/release.yml`。第一次可以完全按本文手动发布，不必配置自动化；以后使用它需要先配置 PyPI Trusted Publishing 和 GitHub 的 `pypi` environment。不要将手动上传和工作流对同一批文件重复执行。

## 常见问题

- **403 / Invalid authentication**：检查 token 属于哪个站点、是否完整、邮箱是否验证、项目是否有上传权限。
- **File already exists**：已用过该文件名；更新版本并重新检查构建，不要反复删除重传。
- **TestPyPI 找不到依赖**：按第 4 步先安装本地 wheel 的正式依赖，第 5 步加 `--no-deps`。
- **无法加载 MODFLOW DLL**：属于原生运行环境问题，pip 安装成功不等于原生求解器路径已经配置。
- **关闭并重开了 PowerShell**：重新设置 `$releasePy`、`$verifyPy`、`$version`、`$distDir`；其中 `$distDir` 指向已检查的那一个构建目录，不能随意选旧包。
