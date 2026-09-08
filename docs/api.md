# Public API and model contracts

`MF6PQC` is the recommended public name; `mf6pqc` is the same class for existing
scripts. `MF6PQC.from_config(config)` accepts the following grouped dataclasses:

| Configuration | Purpose |
|---|---|
| `BackendPaths` | Database, chemistry input, MODFLOW library, workspace and output directory |
| `CellFields` | Temperature (°C), pressure (atm), porosity, saturation, density (kg/L), relative viscosity and diffusion |
| `ChemistryOptions` | Water component, volume/density convention, print mask and signed components |
| `FeedbackOptions` | Porosity/K, density, diffusion, molar volumes and conductivity updater |
| `SIAOptions` | Iterations, finite tolerances, relaxation, strict failure and optional aqueous rate evaluator |
| `EnergyOptions` | Explicit GWE/VSC coupling and ownership controls |
| `OutputOptions` | One-based retained steps, interval, legacy offset and progress interval |

Construction initializes chemistry so component names and initial concentrations
are available before the application's FloPy model is built. MODFLOW initialization
is deferred until `run()`. The [E01 driver](../examples/PHT3D_E01/run.py) shows a
complete working keyword-based workflow, and the
[thermal driver](../examples/GWE_VSC_Reactive/run.py) uses structured configuration.

## Initial and boundary chemistry

```python
initial = simulator.setup({"solution": 0, "equilibrium_phases": 1})
components = simulator.get_components()
inflow = simulator.get_initial_concentrations(1)
```

The seven mapping names are `solution`, `equilibrium_phases`, `exchange`,
`surface`, `gas_phase`, `solid_solutions`, and `kinetics`. Values are PHREEQC
entity numbers: a scalar or one value per cell. `-1` means no entity. Unknown
names, fractional indices, booleans and int32 overflow are rejected.

For mixed initial conditions, pass `ic_map2` and `fractions` together. The
fraction weights **the first map**; each fraction must be in `[0, 1]`.

Physical fields accept a scalar or an array with `nxyz` values, flattened in
MODFLOW cell order. Values must be finite; masks must contain only 0/1. Returned
initial concentrations and cached setup results are independent arrays.

## Coupling methods

| Method | Contract |
|---|---|
| `SNIA` (default) | Transport then chemistry over a complete step |
| `SIA` | Source-based iteration; every GWT component needs a `SRC` package |
| `Strang` | Two equal adjacent TDIS half-steps; one reaction over their sum |
| `ThermalSNIA` | Explicit flow/solute/heat progression with lagged VSC temperature |

`run_SNIA()`, `run_SIA()`, `run_Strang()` and `run_ThermalSNIA()` remain available.
`reaction_steps` is an optional nonempty list of one-based MODFLOW steps supported
only by SNIA. It must include the final step. Explicit `save_steps` must be a
subset of the reaction schedule. Strang output steps count logical pairs.

`save_steps=None` uses interval saving; `save_steps=[]` is invalid. The historical
interval rule is `(zero_based_step + save_interval_offset) % save_interval == 0`.
Set `save_interval_offset=1` to save every Nth completed step; the default offset
0 preserves historical first-step saving. The initial chemical state is always
retained. A final state is saved only when requested by the schedule.

## Feedback and units

All current native workspaces must use `TIME_UNITS DAYS`. Chemistry receives
seconds. Solution concentrations passed to PhreeqcRM use its mol/L setting.
Selected output such as PHREEQC `TOT()` may instead be mol/kg water: explicitly
label each punched quantity in your model documentation. No universal unit is
assigned to all selected-output columns.

Density input is kg/L; BUY receives kg/m³. With calculated-density feedback,
`GetDensityCalculated()` is read independently and never overwrites a selected
output column. With selected-output density, the final heading must match
`density_output_heading` (default `RHO`).

Mineral changes are `d_<mineral>` amounts per representative bulk litre; molar
volumes are L/mol. K means hydraulic conductivity in model length/day, not
intrinsic permeability. The legacy vertical/horizontal ratio defaults to 0.6;
set it explicitly for the model. VSC owns viscosity-adjusted K and cannot be
combined with an updater that applies viscosity again.

Use `KozenyCarmanUpdater`, `PowerLawUpdater`, or subclass
`BaseHydraulicConductivityUpdater` from `mf6pqc.permeability`. Keep case-specific
mineral constants, boundaries and empirical formulas in your model.

## Results and failure behavior

`save_results()` creates `results.npy`, `results_headings.txt`,
`results_times.npy`, and `results_manifest.json`. Porosity and K arrays share the
full time axis. Legacy diffusion arrays contain saved reaction states only and
use `results_times[1:]`. Thermal arrays include both the temperature used by the
flow solve and the post-transport chemistry temperature.

Arrays are validated before writing and replaced atomically one file at a time.
The manifest is the completion marker and is written last; a missing manifest
means the file group is incomplete. Readers should use the manifest's file list,
as unrelated older sidecars may remain in a reused output folder. Use a new
output directory for independently reproducible runs; concurrent writers to the
same result path are unsupported.

Exceptions are defined in `mf6pqc.exceptions`: `ConfigurationError`,
`BackendError`, `CouplingError`, `ConvergenceError`, and `PropertyUpdateError`.
Set `fail_on_nonconvergence=True` and `sia_fail_on_nonconvergence=True` for strict
scientific runs. Diagnostic failures are retained in the manifest. After any
failed run or failed chemistry setup, resources are finalized and a new instance
is required. `finalize()` is idempotent; results can still be saved after a
successful run has been finalized.

## Logging

The package uses Python loggers under `mf6pqc`. Enable progress messages in your application with `logging.basicConfig(level=logging.INFO)`. Example command-line drivers configure logging only when explicitly executed. Native MODFLOW/PhreeqcRM diagnostics may also write their own files or console messages.
