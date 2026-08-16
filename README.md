# MF6PQC

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](pyproject.toml)

MF6PQC is a research-grade reactive-transport framework coupling
[MODFLOW 6](https://www.usgs.gov/software/modflow-version-670) groundwater
flow and transport with
[PhreeqcRM](https://www.usgs.gov/software/phreeqcrm-reaction-module-transport-simulators)
geochemistry.

## Installation

Python 3.11 or newer is required.

```powershell
python -m pip install -e ".[examples]"
```

MODFLOW 6 executables and its shared library are not included in this
repository. The examples expect MODFLOW 6.7.0 under `bin/mf6.7.0/`.

## Run an example

```powershell
python examples\PHT3D_E01\run.py
python examples\PHT3D_E01\validate.py
```

## Run tests

```powershell
python -m compileall -q mf6pqc tests examples
python -m unittest discover -s tests -v
```
