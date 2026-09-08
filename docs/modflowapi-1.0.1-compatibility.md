# modflowapi 1.0.1 compatibility verification

Verified on 2026-09-08 in `C:\ProgramFiles\Miniconda\python.exe` (Python 3.12.9,
Windows). This is a Python compatibility audit, not a new numerical benchmark run.

## Upgrade

- Installed the official PyPI `modflowapi==1.0.1` wheel, replacing 0.2.0 in the
  active environment. Verified its SHA-256 against PyPI metadata:
  `4c997cc001a4c67a5121367245a3ca4edf4588cd39f9a16f9c0e966f0a60286a`.
- Updated the single dependency declaration to `modflowapi>=1.0.1,<2`.
  `requirements.txt` already delegates to `pyproject.toml`.
- NumPy 2.2.1, pandas 2.2.3 and xmipy 1.5.0 meet the new requirements;
  no changes to those packages, PhreeqcRM 0.0.18 or MODFLOW shared libraries.
- No coupling algorithms or example model parameters needed modification.

The [official 1.0.0 release notes](https://github.com/MODFLOW-ORG/modflowapi/releases/tag/1.0.0)
state that the Python interface is unchanged from 0.2.0 and now follows semantic
versioning, with Python >=3.11 required. The
[1.0.1 release](https://github.com/MODFLOW-ORG/modflowapi/releases/tag/1.0.1)
raises the pandas requirement to support NumPy 2. The installed wheel requires
pandas >=2.2.2 and xmipy >=1.5.0.

## Results without model execution

| Check | Result |
|---|---|
| `python -m unittest discover -s tests -q` | 76 tests passed, approximately 1 second |
| `python scripts/check_examples.py --allow-notebook-outputs` | All 22 cases passed |
| Ruff 0.16.0 lint, `mf6pqc tests examples scripts` | Passed |
| Ruff 0.16.0 format check, same paths | 115 files passed |
| Core and modflowapi dependency versions | All satisfy declared requirements |

The three new tests in `tests/test_modflowapi_compatibility.py` check the real
ModflowApi constructor against installed xmipy (intercepting native creation),
bind coupling call signatures to the installed API, and load real ApiSimulation,
ApiModel and array wrappers against an in-memory variable store. They check
GWF/GWT/GWE discovery, solution lookup, geometry and live concentration updates.
They do not load a native solver or solve a model.

Case checks cover directory structure, plotting-notebook syntax and Python
imports in a fresh process for each case. Both native factory methods are
blocked during imports. No `run.py` main function, native integration profile,
model initialization, time step, chemical calculation or notebook cell was run.
Existing simulation inputs and numerical outputs were not regenerated.

Cases: GWE_VSC_Reactive, Hamann2015, PHT3D_E01 through PHT3D_E13,
SaltLake_Brine3D, Splitting_KineticDecay, Splitting_RedoxFront2D and
Xie2015_B1 through Xie2015_B4.

## Scope and existing environment observations

- E02 and E13 plotting notebooks contain saved execution output. The new optional
  `--allow-notebook-outputs` flag preserves it and still checks code syntax and
  imports. The default release-cleanliness check continues to reject saved output.
- JupyterLab is absent from this environment; the example runtime dependencies
  are installed and meet their version constraints. Launching JupyterLab itself
  requires installing the examples extra. No notebook frontend was installed by
  this dependency-only upgrade.
- The global `pip check` reports an unrelated existing issue: pyvista 0.48.4
  requires the missing cyclopts package. Neither package is a dependency of
  MF6PQC or modflowapi; this upgrade does not change them.
- No end-to-end numerical equivalence, solver convergence or completion of the
  long examples is claimed. Those require actual native model runs, which the
  user explicitly excluded. Earlier 0.2.0 numerical verification records remain
  historical records and have not been relabeled as 1.0.1 results.
