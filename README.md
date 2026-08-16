# MF6PQC

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

MF6PQC is a research-grade reactive-transport framework coupling
[MODFLOW 6](https://www.usgs.gov/software/modflow-version-670) groundwater
flow and transport with
[PhreeqcRM](https://www.usgs.gov/software/phreeqcrm-reaction-module-transport-simulators)
geochemistry.

It is designed for hydrogeological research in which flow, multispecies
transport, reactions, and selected medium properties evolve together.

## Capabilities

- Independent MODFLOW 6 GWT models for PhreeqcRM transport components.
- Equilibrium, kinetic, exchange, surface, gas, and solid-solution chemistry
  through PhreeqcRM.
- SNIA, source-based SIA, and symmetric Strang coupling strategies.
- Optional porosity, NPF hydraulic-conductivity, diffusion, and BUY density
  feedback.
- Opt-in MODFLOW 6 GWE energy transport, VSC viscosity feedback, and
  cell-wise GWE-temperature synchronization into PhreeqcRM reactions.
- Replaceable hydraulic-property models, including a stable interface for
  future learned models.
- Explicit result times, atomic NumPy output, a manifest, convergence records,
  and quantitative public validation cases.

## Architecture

```text
scientific case / configuration
             |
          MF6PQC facade
             |
     coupling algorithm registry
    /        |       |          \
 SNIA      SIA    Strang    ThermalSNIA
    \        |       |          /
        shared coupling primitives
          /                 \
  MODFLOW 6 + energy     PhreeqcRM backend
             |
   hydrogeological feedback models
             |
     validated result serialization
```

The simulator facade preserves the user workflow. Numerical ownership is
separated into `mf6pqc/coupling/`, native-library lifecycle into
`mf6pqc/backends.py`, thermal pointer/property ownership into
`mf6pqc/energy.py`, and reaction-driven property evolution into
`mf6pqc/feedback.py` and `mf6pqc/permeability.py`. See the
[architecture guide](docs/architecture.md), [scientific contracts](docs/scientific-contracts.md),
[splitting audit](docs/splitting-audit.md), and
[thermal coupling contract](docs/thermal-coupling.md).

## Installation

Python 3.11 or newer is required by current PhreeqcRM wheels.

```powershell
python -m pip install -e ".[examples]"
```

MODFLOW 6 executables and its shared library are not bundled. The examples in
this repository expect MODFLOW 6.7.0 under `bin/mf6.7.0/`.

## A structured case configuration

```python
from mf6pqc import BackendPaths, MF6PQC, SimulationConfig

config = SimulationConfig(
    case_name="my_case",
    nxyz=150,
    paths=BackendPaths(
        database="input/phreeqc.dat",
        chemistry_input="input/model.pqi",
        modflow_library="bin/mf6.7.0/libmf6.dll",
        workspace="simulation",
        output_directory="output",
    ),
)

with MF6PQC.from_config(config) as simulator:
    initial = simulator.setup({"solution": 0})
    components = simulator.get_components()
    # Build and write the matching GWF/GWT simulation here.
    simulator.run(method="SNIA")
    simulator.save_results()
```

Existing flat dictionaries remain valid:

```python
from mf6pqc import mf6pqc

simulator = mf6pqc(**sim_params)
initial = simulator.setup(ic_map=ic_mapping)
simulator.run()  # historical alias for SNIA
simulator.save_results()
simulator.finalize()
```

## Coupling strategies

| Method | Operator sequence | Intended use |
|---|---|---|
| SNIA | transport → reaction → feedback | Robust default and compatibility path |
| SIA | repeated transport/source correction ↔ reaction | Strong within-step coupling when every residual converges |
| Strang | transport(dt/2) → reaction(dt) → transport(dt/2) | Reduced temporal splitting error; requires equal TDIS step pairs |
| ThermalSNIA | GWF/GWT/GWE/VSC → temperature-aware reaction → feedback | Explicit opt-in thermal/viscosity coupling |

Use `run()`, `run_SNIA()`, `run_SIA()`, `run_Strang()`,
`run_ThermalSNIA()`, or `run(method="...")`. `run()` remains ordinary SNIA;
thermal cases must explicitly select `ThermalSNIA`. Medium-property feedback
is committed after reaction for SNIA, after convergence for SIA, and at the
reaction midpoint before Strang's second transport half-step. It is not
iterated inside the current SIA source Picard loop.

## Fast verification

The normal development loop does not run the long Xie2015 cases.

```powershell
python -m compileall -q mf6pqc tests examples
python -m unittest discover -s tests -v
python examples\PHT3D_E01\run.py
python examples\PHT3D_E01\validate.py
python examples\Splitting_KineticDecay\reaction_only_check.py
python examples\GWE_VSC_Reactive\run.py
python examples\GWE_VSC_Reactive\validate.py
```

PHT3D_E01 takes only a few seconds on the development machine. The
reaction-only splitting invariant detects repeated advancement of saved
kinetic state. The full Steefel-1996 one-dimensional replication and the
two-dimensional `Splitting_RedoxFront2D` comparison belong to the extended
validation tier; both use `plot.ipynb` for figures.
Xie2015 and other long cases belong
to the deliberate release-validation tier described in
[the validation guide](docs/validation.md).

## Results

`save_results()` writes compatible NumPy arrays and additional reproducibility
metadata:

- `results.npy`: selected output with shape `(time, output, cell)`;
- `results_headings.txt`: selected-output names;
- `results_times.npy`: simulation time for every selected-output frame;
- `results_manifest.json`: schema, files, method, steps, wall time, and
  convergence records;
- optional `results_porosity.npy`, `results_K.npy`, and `results_diffc.npy`;
- thermal runs additionally write temperature, temperature-used-for-flow,
  viscosity, reference-K, and effective-K arrays.

## Scientific cautions

- MODFLOW NPF `K` is **hydraulic conductivity**, not intrinsic permeability.
- BUY already represents variable-density flow. The optional fluid-adjusted
  K updater must not be combined blindly with BUY/VSC because fluid effects
  may be counted twice.
- With VSC, reaction feedback owns only NPF `KxxINPUT` (reference K); MODFLOW
  VSC exclusively owns viscosity-adjusted `Kxx` (effective K). MF6PQC rejects
  the fluid-adjusted K updater and manual boundary-conductance feedback in
  this mode.
- `d_<mineral>` porosity feedback assumes mineral amounts and molar volumes
  form a bulk-volume fraction. Molar volumes are model input in L/mol and can
  be overridden per case.
- `d0` must use the squared MODFLOW length unit per MODFLOW time unit.
- Static TDIS schedules are supported; adaptive time stepping requires a
  dedicated coupling contract before it should be used.
- SIA currently computes mobile water volume from configured porosity and
  saturation; transient `FMI/GWFSAT` synchronization is not yet a validated
  unsaturated-flow feature.
- Source-based SIA requires a GWT package named `SRC`. Strongly generated
  mobile daughters can make the algebraic transport-residual intermediate
  negative; use strict convergence and a step-size study rather than treating
  domain clipping as a global-implicit solution.

## Roadmap

The architecture reserves clear extension points, but the following are not
claimed as implemented features yet:

1. additional iterative and higher-order coupling algorithms;
2. trained, constrained hydrogeological-property models with provenance and
   out-of-distribution safeguards.

