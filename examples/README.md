# Example catalog

The source archive includes 22 examples. Install the `examples` extra and
configure the native libraries as described in [installation](../docs/installation.md).
Examples and reference data are not installed by the wheel.

Each example has `input_data/`, `modflow_model.py`, `run.py` and `plot.ipynb`.
`simulation/` and `output/` hold generated files; only their `.gitkeep` markers
are distributed. `validate.py` and `plot.py` are optional. Importing a script
does not run a model. Clear notebook execution output before distributing it.

| Example | Standalone validator | Execution scope |
|---|---|---|
| [GWE_VSC_Reactive](GWE_VSC_Reactive/run.py) | [validate.py](GWE_VSC_Reactive/validate.py) | Run explicitly; cost depends on configuration |
| [Hamann2015](Hamann2015/run.py) | No; analysis is in the notebook or driver | Long benchmark; run separately |
| [PHT3D_E01](PHT3D_E01/run.py) | [validate.py](PHT3D_E01/validate.py) | Run explicitly; cost depends on configuration |
| [PHT3D_E02](PHT3D_E02/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E03](PHT3D_E03/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E04](PHT3D_E04/run.py) | [validate.py](PHT3D_E04/validate.py) | Run explicitly; cost depends on configuration |
| [PHT3D_E05](PHT3D_E05/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E06](PHT3D_E06/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E07](PHT3D_E07/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E08](PHT3D_E08/run.py) | [validate.py](PHT3D_E08/validate.py) | Run explicitly; cost depends on configuration |
| [PHT3D_E09](PHT3D_E09/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E10](PHT3D_E10/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E11](PHT3D_E11/run.py) | [validate.py](PHT3D_E11/validate.py) | Long benchmark; run separately |
| [PHT3D_E12](PHT3D_E12/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [PHT3D_E13](PHT3D_E13/run.py) | [validate.py](PHT3D_E13/validate.py) | Known pH/Ca reference discrepancy |
| [SaltLake_Brine3D](SaltLake_Brine3D/run.py) | [validate.py](SaltLake_Brine3D/validate.py) | Short `smoke` profile; long `highres` profile |
| [Splitting_KineticDecay](Splitting_KineticDecay/run.py) | [validate.py](Splitting_KineticDecay/validate.py) | Run explicitly; cost depends on configuration |
| [Splitting_RedoxFront2D](Splitting_RedoxFront2D/run.py) | No; analysis is in the notebook or driver | Run explicitly; cost depends on configuration |
| [Xie2015_B1](Xie2015_B1/run.py) | No; analysis is in the notebook or driver | Long benchmark; run separately |
| [Xie2015_B2](Xie2015_B2/run.py) | No; analysis is in the notebook or driver | Long benchmark; run separately |
| [Xie2015_B3](Xie2015_B3/run.py) | No; analysis is in the notebook or driver | Long benchmark; run separately |
| [Xie2015_B4](Xie2015_B4/run.py) | No; analysis is in the notebook or driver | Long benchmark; run separately |

The presence of a validator is not a claim that a case passed for the current
version. See [verification history](../docs/release-readiness.md) and the
[validation strategy](../docs/validation.md). Native checks run only with an
explicit `--native` option to `scripts/check_examples.py`; ordinary CI checks
syntax and guarded imports without running the models.
