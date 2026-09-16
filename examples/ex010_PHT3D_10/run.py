"""Run the ex010_PHT3D_10 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex010_PHT3D_10.modflow_model import build_model
from mf6pqc import MF6PQC, BackendPaths, CellFields, ProcessBackendFactory, SimulationConfig

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
NROW = 40
NCOL = 80
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.3
PERLEN = 500
NSTP = 50
LENGTH_X = 200.0
LENGTH_Y = 50.0
TOP = 10.0
BOTM = 0.0
TSMULT = 1.0
ALH = 0.5
ATH1 = 0.1
DIFFC = 3e-10
INLET_HEAD = 5.0
OUTLET_HEAD = 3.0


def main() -> None:
    kinetics_mask = np.load(INPUT_DIR / "benzene_napl.npy", allow_pickle=False).ravel()
    kinetics_mask[kinetics_mask == 0.2] = 1
    kinetics_mask = kinetics_mask.astype(int)
    ic_mapping = {"solution": 0, "equilibrium_phases": 1, "kinetics": kinetics_mask}
    hydraulic_conductivity = np.load(INPUT_DIR / "hk.npy", allow_pickle=False).reshape(
        NLAY, NROW, NCOL
    )
    initial_head = np.load(INPUT_DIR / "head.npy", allow_pickle=False).reshape(NLAY, NROW, NCOL)
    simulation_config = SimulationConfig(
        case_name="ex010",
        nxyz=NXYZ,
        nthreads=1,
        backend_factory=ProcessBackendFactory(processes=min(16, os.cpu_count() or 1)),
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(pressure_atm=1.0, porosity=POROSITY),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(0)
        build_model(
            workspace=WORKSPACE,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            length_x=LENGTH_X,
            length_y=LENGTH_Y,
            top=TOP,
            botm=BOTM,
            perlen=PERLEN,
            nstp=NSTP,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            initial_head=initial_head,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            tsmult=TSMULT,
        )
        simulator.run()
        simulator.save_results()
        print("ex010 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
