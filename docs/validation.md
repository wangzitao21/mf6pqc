# Validation strategy

MF6PQC uses tiered validation so ordinary refactoring remains fast while long
scientific cases are still protected before a release.

## Tier 0 — syntax and import checks

Target: seconds.

```powershell
python -m compileall -q mf6pqc tests examples
```

## Tier 1 — backend-free regression tests

Target: less than one second. These tests use small fake backend objects and
cover array layout, time schedules, initial-condition maps, constitutive
relationships, result serialization, convergence policy, and lifecycle.

```powershell
python -m unittest discover -s tests -v
```

## Tier 2 — short native integration

Target: seconds to tens of seconds.

```powershell
python examples\PHT3D_E01\run.py
python examples\PHT3D_E01\validate.py
python examples\Splitting_KineticDecay\reaction_only_check.py
python examples\GWE_VSC_Reactive\run.py
python examples\GWE_VSC_Reactive\validate.py
```

PHT3D_E01 protects the historical SNIA path against official PHT3D data. The
kinetic-decay reaction-only check requires SNIA, SIA, and Strang to advance one
saved kinetic state exactly once. GWE_VSC_Reactive checks the exact VSC constitutive
identity, GWE-to-PhreeqcRM temperature synchronization, Arrhenius kinetics,
reaction-driven porosity/reference-K feedback, and a solved seepage field.

## Tier 3 — selected public examples

Target: minutes. Run cases chosen to exercise distinct chemistry rather than
every similar notebook. A release candidate should include at least one case
for equilibrium phases, kinetics, surface/exchange, two-dimensional transport,
porosity/K feedback, and density feedback.

For changes to coupling algorithms, run both extended splitting studies:

```powershell
python examples\Splitting_KineticDecay\run.py
python examples\Splitting_RedoxFront2D\run.py
```

The first reproduces Steefel and MacQuarrie's equation (114) and Figure 6 at
CFL 0.1, 0.5, and 1 against an analytical solution. It reports both the
paper-node and MODFLOW cell-center coordinate conventions and requires the
CFL-1 paper-node error to order as SIA < Strang < SNIA.
The second is a stiff, capacity-limited two-dimensional redox-front stress
test. Its predeclared combined Don/solid-extent error must order as
SIA < Strang < SNIA, while deterministic transport work must order as
SIA > Strang > SNIA. It uses a 0.125-day Strang reference, a 0.25-day
time-refinement check, and an independent 0.125-day SNIA cross-method check.
Trial SIA solves are work evaluations, not physical time advances. This
strong-splitting result is a regression contract for the named case, not a
universal ranking for smooth reactions or refined time steps.

Both simulation scripts write data and metrics only. Execute the corresponding
`plot.ipynb` notebooks after quantitative validation to generate figures;
plot inspection is not a validator.

Validators must compare named quantities at named times using documented
absolute or relative tolerances. Plot inspection alone is not a pass criterion.

## Tier 4 — long release benchmarks

Target: hours. Xie2015 and other long cases are never part of the ordinary
edit-test loop. Run them only when changes touch their numerical path or for a
release candidate. Record:

- code commit and dirty-tree status;
- MODFLOW and Python package versions;
- wall time, cell count, step count, and thread count;
- result manifest and file checksums;
- quantitative error metrics against the public reference;
- convergence-failure records and physical-range checks.

For long runs, validate one representative case first. Do not launch all cases
concurrently unless memory, native-library thread use, and output paths have
been reviewed.

## Reference-data policy

Reference arrays must identify their source, extraction method, units, spatial
ordering, and output time. Regenerating a reference is a scientific change and
must not be hidden inside a code refactor. Store compact immutable reference
data; do not commit routine simulation workspaces or generated output.

## Numerical comparison policy

Bitwise equality is valuable for a fixed machine and backend version, but is
not the sole release criterion. Prefer a combination of:

- shape and finite-value checks;
- mass/water balance;
- maximum absolute error;
- RMSE or normalized RMSE;
- expected ordering or monotonicity;
- physically meaningful bounds;
- bitwise/hash checks only for tightly controlled baselines.
