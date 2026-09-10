# modflowapi 1.0.1 compatibility

The compatibility check recorded on 2026-09-08 used Windows and Python 3.12.9.
MF6PQC requires `modflowapi>=1.0.1,<2`. No coupling algorithm or example model
parameter was changed for this dependency update.

The [official 1.0.0 release notes](https://github.com/MODFLOW-ORG/modflowapi/releases/tag/1.0.0)
state that the Python interface is unchanged from modflowapi 0.2.0 and requires
Python 3.11 or later. The
[1.0.1 release](https://github.com/MODFLOW-ORG/modflowapi/releases/tag/1.0.1)
raises the pandas requirement to support NumPy 2. Its wheel requires
pandas >=2.2.2 and xmipy >=1.5.0.

## Recorded checks

| Check | Outcome |
|---|---|
| Backend-free unit tests | 76 passed |
| Example layout, notebook syntax and guarded imports | 22 passed |
| Ruff 0.16.0 lint and format | Passed |
| Runtime dependency versions | Satisfied the declared requirements |

`tests/test_modflowapi_compatibility.py` checks the real ModflowApi constructor
against xmipy while intercepting native creation, binds coupling call signatures
to the installed API, and loads real extension wrappers against an in-memory
variable store. It covers GWF/GWT/GWE discovery, solution lookup, geometry and
live concentration updates.

The example import check blocks native backend construction. No native model,
time step or plotting-notebook cell was executed. Saved notebook output was
allowed in that historical check; distribution checks require cleared output.

These results establish Python-interface compatibility. They do not establish
numerical equivalence, solver convergence or completion of long examples. The
[numerical verification history](release-readiness.md) retains its original
backend versions.
