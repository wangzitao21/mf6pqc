# Installation and native solvers

Install MF6PQC from a checkout with `python -m pip install -e .`, or install a
built wheel with `python -m pip install dist/mf6pqc-<version>-py3-none-any.whl`.
Use `.[examples]` for FloPy, SciPy, plotting and Jupyter, and `.[dev,examples]`
for the complete development environment. `requirements.txt` delegates to the
same package extras so dependency declarations cannot drift.

The Python wheel contains the coupling package and type marker. The source
archive additionally includes documentation, tests and example inputs. It
excludes native binaries, generated model workspaces/results, and local research
collections. Installing a wheel does not install the source-tree examples.

## Python API compatibility

MF6PQC requires `modflowapi>=1.0.1,<2` and Python 3.11 or later.
modflowapi 1.0.1 requires pandas 2.2.2 or later; pip resolves this transitive
requirement even when the examples extra is not selected. Upgrade an existing
checkout environment with `python -m pip install --upgrade -e .`.

The [1.0.1 compatibility check](modflowapi-1.0.1-compatibility.md) covers imports,
Python interfaces and backend-free regressions without executing example models.

## MODFLOW 6

Obtain a MODFLOW 6 shared library from the
[official MODFLOW releases](https://github.com/MODFLOW-USGS/modflow6/releases).
The process architecture must match Python (normally 64 bit). MODFLOW 6.7.0
and 6.8.0 are used by the existing Windows examples; changing a backend version
requires rerunning the relevant quantitative checks.

Supply the full library path as `BackendPaths.modflow_library` or
`modflow_dll_path`. The example helper supports these environment variables:

| Variable | Meaning |
|---|---|
| `MF6PQC_LIBMF6` | Absolute shared-library path; takes precedence over `MF6PQC_BIN` |
| `MF6PQC_BIN` | Directory containing `libmf6.dll`, `libmf6.so`, or `libmf6.dylib` |
| `MF6PQC_MF6_EXE` | Optional executable path used in FloPy simulation metadata |
| `MF6PQC_RUN_ROOT` | Optional root for generated `<case>/output` and `<case>/simulation` |
| `MF6PQC_USE_INSTALLED=1` | Test an installed wheel instead of importing the checkout |

These convenience variables belong to the examples. The runtime package uses
the explicit paths supplied by your application and has no repository-layout
assumptions.

## PhreeqcRM

The `phreeqcrm` Python distribution supplies the native chemistry binding.
MF6PQC loads it lazily when constructing a simulator. Importing `mf6pqc` alone
does not load MODFLOW or PhreeqcRM. Chemistry databases and chemical-model input
files are scientific inputs supplied by the application; they are not silently
selected or downloaded by the package.

## Troubleshooting

- **Missing DLL/shared object:** check the path, architecture, native dependencies,
  and library version. The executable and shared library are different files.
- **Missing component GWT model:** build every model using the exact component
  list returned by `get_components()` and `get_gwt_model_name()`.
- **TDIS error:** use `TIME_UNITS DAYS` and a static schedule. Other units and ATS
  have no supported coupling contract in this release.
- **PhreeqcRM failure:** negative native status codes become `BackendError` with
  the operation and native message. Check database/input compatibility and logs.
- **Nonconvergence:** enable strict MODFLOW/SIA failure options, refine the input
  time/grid discretization, and rerun from a fresh simulator.
- **Windows file replacement failure:** close notebook memory maps or applications
  holding the output open, or select a new `MF6PQC_RUN_ROOT`.

An initialized native solver is stateful. Always use a context manager or
`try/finally: simulator.finalize()`. Failed or finalized instances cannot be reused.
