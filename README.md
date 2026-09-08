# MF6PQC

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

MF6PQC couples MODFLOW 6 flow and solute transport with PhreeqcRM chemistry.
It supports sequential non-iterative (SNIA), sequential iterative (SIA), and
Strang splitting, with optional reaction-driven porosity, hydraulic
conductivity, diffusion, and density feedback. An opt-in thermal pathway
couples MODFLOW GWE/VSC temperature and viscosity with chemistry.

This is research software. A converged calculation and a validated conceptual
model are separate requirements. See the [scientific contracts](docs/scientific-contracts.md)
and the [release verification record](docs/release-readiness.md) for tested
behavior and known limitations. The current version remains a development
release while outstanding benchmark discrepancies are investigated.

## Installation

Python 3.11 or newer is required. From a source checkout:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[examples]"
```

After the first PyPI release, installation will be `python -m pip install mf6pqc`.
The core dependencies are NumPy, modflowapi, and phreeqcrm. FloPy, plotting,
notebook, and example-specific dependencies are in the `examples` extra.

MODFLOW's shared library is installed separately; a standalone `mf6` executable
is insufficient for API coupling. The package does not redistribute native
MODFLOW binaries. The source examples use `bin/mf6.7.0/` or `bin/mf6.8.0/`
according to their existing benchmark configuration. To use another installation:

```powershell
$env:MF6PQC_BIN = "C:\tools\modflow6"
# Or specify the full shared-library path:
$env:MF6PQC_LIBMF6 = "C:\tools\modflow6\libmf6.dll"
```

On Linux use `export MF6PQC_BIN=/path/to/modflow6`; the helper selects
`libmf6.so` (`libmf6.dylib` on macOS). Native-library availability and
compatibility must be checked on the target platform. See
[installation and troubleshooting](docs/installation.md).

## First example

```bash
python examples/PHT3D_E01/run.py
python examples/PHT3D_E01/validate.py
jupyter lab examples/PHT3D_E01/plot.ipynb
```

Each case contains `input_data/`, `simulation/`, `output/`, `modflow_model.py`,
`run.py`, and `plot.ipynb`. Optional `validate.py` and `plot.py` contain quantitative
checks and reusable plotting functions. Paths are anchored to the case; commands
can also be launched from another working directory using an absolute script path.
Importing an example does not start a simulation.

For an integrated porosity/density/K example:

```bash
python examples/SaltLake_Brine3D/run.py --profile smoke --scenario feedback
python examples/SaltLake_Brine3D/validate.py --profile smoke --scenario feedback
```

The [example catalog](examples/README.md) identifies slow cases and available
validators. Long benchmarks are deliberately excluded from automated checks.

## Python API

Use `MF6PQC` and `SimulationConfig` for new applications. The historical
`mf6pqc` class name and keyword constructor remain supported.

1. Define a PHREEQC database, chemical input, and initial-condition map.
2. Construct the simulator and call `setup()` to obtain component concentrations.
3. Build a MODFLOW simulation with one GWT model for each returned component.
4. Call `run(method="SNIA")`, then `save_results()` inside a context manager.

See [the API guide](docs/api.md), the working
[E01 driver](examples/PHT3D_E01/run.py), and [architecture](docs/architecture.md).
All current coupling paths require **TDIS time units DAYS**. PhreeqcRM receives
seconds. Unsupported time units and ATS are rejected before native initialization.

Results include selected-output headings, the actual retained time axis, optional
physical-property arrays, and a JSON manifest with solver settings, software
versions, and input/library fingerprints. Concentrations use component-major
ordering; result arrays use `(time, selected_output, cell)`.

## Development and verification

```bash
python -m pip install -e ".[dev,examples]"
python -m unittest discover -s tests -v
python -m ruff check mf6pqc tests examples scripts
python -m ruff format --check mf6pqc tests examples scripts
python scripts/check_examples.py
python scripts/check_examples.py --native PHT3D_E01 GWE_VSC_Reactive
python -m build
python -m twine check --strict dist/*
```

The native-check tool uses an explicit short-case allowlist and isolated output
folders. It cannot run Xie2015 B1–B4, PHT3D E11, or Hamann2015.
See [validation](docs/validation.md), [contributing](CONTRIBUTING.md), and
[release procedure](docs/releasing.md).

## License and attribution

MF6PQC is distributed under GPL-3.0-only. MODFLOW 6, PhreeqcRM, their databases,
and the cited benchmark studies remain independently authored software/data.
Retain source attribution when reusing reference data. The software citation
metadata is in [CITATION.cff](CITATION.cff); cite the original solver and
benchmark publications as appropriate for your study.
