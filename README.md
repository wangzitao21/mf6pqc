# MF6PQC v1.0

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

MF6PQC couples MODFLOW 6 flow and solute transport with PhreeqcRM chemistry.
It supports sequential non-iterative (SNIA), sequential iterative (SIA), and
Strang splitting, with optional reaction-driven porosity, hydraulic
conductivity, diffusion, and density feedback. An opt-in thermal pathway
couples MODFLOW GWE/VSC temperature and viscosity with chemistry.

## Installation

Python 3.11 or newer is required. Install from PyPI with `python -m pip install mf6pqc`.

The core dependencies are NumPy, modflowapi, and phreeqcrm. FloPy, plotting,
notebook, and example-specific dependencies are in the `examples` extra.

MODFLOW's shared library is installed separately.

## First example

```bash
python examples/PHT3D_E01/run.py
python examples/PHT3D_E01/validate.py
jupyter lab examples/PHT3D_E01/plot.ipynb
```

Each case contains `input_data/`, `simulation/`, `output/`, `modflow_model.py`,
`run.py`, and `plot.ipynb`.

## License and attribution

MF6PQC is distributed under GPL-3.0-only. MODFLOW 6, PhreeqcRM, their databases,
and the cited benchmark studies remain independently authored software/data.
Retain source attribution when reusing reference data; cite the original solver
and benchmark publications as appropriate for your study.
