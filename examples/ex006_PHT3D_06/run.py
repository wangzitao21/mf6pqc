"""Run the ex006_PHT3D_06 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex006_PHT3D_06.modflow_model import build_model
from mf6pqc import MF6PQC, BackendPaths, CellFields, SimulationConfig

INPUT_DIR = CASE_DIR / "input_data"
WORKSPACE = CASE_DIR / "simulation"
OUTPUT_DIR = CASE_DIR / "output"
MODFLOW_LIBRARY = Path(
    os.environ.get(
        "MF6PQC_LIBMF6",
        CASE_DIR.parents[1]
        / "bin"
        / "mf6.8.0"
        / {"win32": "libmf6.dll", "darwin": "libmf6.dylib"}.get(sys.platform, "libmf6.so"),
    )
)

NLAY = 1
NROW = 1
NCOL = 180
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.424
DELR = [0.2 / NCOL] * NCOL
DELC = [1.0]
TOP = 1.0
BOTM = 0.0
HYDRAULIC_CONDUCTIVITY = 1.0
PERLEN = np.array([0.05944444, 0.1488889])
NSTP = np.array([250, 600])
TSMULT = np.array([1.0, 1.0])
ALH = 0.000576
ATH1 = 5.76e-05
DIFFC = 0.0
INITIAL_HEAD = 1.0
OUTLET_HEAD = 1.0
INFLOW_RATE = 2.839
PERIOD_DATA = list(zip(PERLEN, NSTP, TSMULT, strict=True))


def main() -> None:
    ic_mapping = {"solution": 0, "equilibrium_phases": 1, "exchange": 1}
    simulation_config = SimulationConfig(
        case_name="ex006",
        nxyz=NXYZ,
        nthreads=6,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(porosity=POROSITY),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        boundary = {
            "H": 110.684176,
            "O": 55.3421163,
            "Charge": 3.79645865e-16,
            "C": 1.42e-05,
            "Ca": 0.0,
            "Cl": 0.0,
            "Na": 0.005,
            "T": 0.005,
        }
        pulse_concentrations = np.array([boundary[name] for name in species])
        boundary.update(Cl=0.005, T=0.0)
        chase_concentrations = np.array([boundary[name] for name in species])
        build_model(
            workspace=WORKSPACE,
            species=species,
            initial_concentrations=initial_concentrations,
            pulse_concentrations=pulse_concentrations,
            chase_concentrations=chase_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            period_data=PERIOD_DATA,
            porosity=POROSITY,
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            initial_head=INITIAL_HEAD,
            outlet_head=OUTLET_HEAD,
            inflow_rate=INFLOW_RATE,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        print("ex006 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
