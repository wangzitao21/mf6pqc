# MF6PQC

**MF6PQC** is a framework that couples [MODFLOW6](https://github.com/MODFLOW-ORG/modflow6) with [PHREEQCRM](https://github.com/usgs-coupled/phreeqcrm), designed for simulating groundwater flow, hydrochemical processes, and aquifer parameter evolution.

---

## Installation & Setup

1. Download the appropriate `mf6` executable for your operating system from [MODFLOW EXECUTABLES](https://github.com/MODFLOW-ORG/executables), and place it in the `./bin/` directory of this project.
2. It is recommended to use a dedicated `conda` environment:

```bash
conda create -n mf6pqc python=3.10
conda activate mf6pqc
pip install -r requirements.txt
````

---

## Usage

1. Navigate to the `examples/` folder and select a case study (e.g., `example1`).
2. Run the simulation:

```bash
python run.py
```

Simulation results will be saved in the corresponding `output` folder.

> ⚠️ Note: Some examples are experimental or under modification and may not run successfully.

---

## Notes

- This project is primarily developed for the author’s research and is **not intended as general-purpose open-source software**. Some functions are experimental, and certain examples may be incomplete.
- If issues arise, please refer to the official documentation of **MODFLOW 6** and **PHREEQCRM**, or modify the code according to your research needs.