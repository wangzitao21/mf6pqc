# Verification history and limitations

This document summarizes earlier checks retained in the repository. It is not a
new numerical validation of MF6PQC 1.0.0. The machine-readable record is
[release-verification.json](release-verification.json); its date, environment and
reported results are preserved as historical evidence.

## Numerical checks recorded on 2026-09-07

The recorded environment was Windows, Python 3.12.9, NumPy 2.2.1,
modflowapi 0.2.0, PhreeqcRM 0.0.18 and FloPy 3.10.0, with MODFLOW 6.7.0 or 6.8.0
as selected by each example. The base commit was `a29ebfe`.

| Check | Recorded outcome |
|---|---|
| Backend-free unit tests | 73 passed |
| Example layout, notebook syntax and guarded imports | 22 passed |
| PHT3D_E01 | Reference comparison passed; results and time axis matched the previous core bitwise |
| GWE_VSC_Reactive | Temperature, viscosity, chemistry and porosity/K feedback checks passed |
| PHT3D_E04 | Reference comparison passed; component NRMSE approximately 1.04–1.81% |
| PHT3D_E08 | Four-component reference comparison passed |
| Splitting_KineticDecay, reaction-only | Maximum pairwise method difference 2.02 × 10⁻¹²; maximum analytical error 8.51 × 10⁻⁷ |
| SaltLake_Brine3D, smoke feedback | 252 cells, two years; balance and feedback checks passed |
| Wheel and source distribution | Build, metadata, contents and isolated installation checks passed |

These records do not establish that the current version or a different backend
combination reproduces every benchmark. Detailed temporary logs and baseline
outputs from those checks are not included in the distribution.

## E13 reference-comparison discrepancy

| Quantity | Recorded NRMSE | Acceptance limit |
|---|---:|---:|
| pH | 0.067094 | 0.040 |
| Ca | 0.090854 | 0.045 |

The record reports bitwise-identical E13 outputs for the earlier and revised
cores. Both exceeded the existing pH/Ca thresholds. The reference inputs,
database, sampling times and model conventions need reconciliation before this
comparison can be described as passed. No threshold or reference array was
changed during the v1.0.0 release preparation.

## Coverage limits

The 2026-09-07 record did not rerun Xie2015 B1–B4, PHT3D_E11 or Hamann2015.
It also excluded the complete kinetic/redox splitting studies and native checks
on Linux/macOS or Python 3.11/3.13. Cross-platform CI was configured but not
reported as executed remotely.

The subsequent [modflowapi 1.0.1 compatibility checks](modflowapi-1.0.1-compatibility.md)
cover Python interfaces and tests without native model execution. They do not
replace the earlier numerical environment with modflowapi 1.0.1 in the history.

For a paper, associate each reported result with the actual source version,
backend versions, inputs and analysis scripts used. See the
[validation strategy](validation.md) for the quantitative checks.
