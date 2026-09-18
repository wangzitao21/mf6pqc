"""Run the ex007_PHT3D_07 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex007_PHT3D_07.modflow_model import build_model
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
NCOL = 41
NXYZ = NLAY * NROW * NCOL
POROSITY = 1.0
PERLEN = 40
NSTP = 120
DELR = [1] * NCOL
DELC = [1.0]
TOP = 1.0
BOTM = 0.0
HYDRAULIC_CONDUCTIVITY = 100.0
ALH = 10.0
ATH1 = 1.0
DIFFC = 0.0
INITIAL_HEAD = 1.0
OUTLET_HEAD = 1.0
INFLOW_RATE = 0.4


def main() -> None:
    solution_map = np.zeros(NXYZ, dtype=int)
    solution_map[0] = 1
    kinetics_map = np.ones(NXYZ, dtype=int)
    kinetics_map[0] = -1
    ic_mapping = {"solution": solution_map, "kinetics": kinetics_map}
    simulation_config = SimulationConfig(
        case_name="ex007",
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
        inflow_concentrations = simulator.get_initial_concentrations(1)
        s_a_index = species.index("S_a")
        initial_concentrations[s_a_index * NXYZ] = 0.001
        inflow_concentrations[s_a_index] = 0.001
        build_model(
            workspace=WORKSPACE,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            perlen=PERLEN,
            nstp=NSTP,
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
        print("ex007 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
