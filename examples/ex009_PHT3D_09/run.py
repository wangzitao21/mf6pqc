"""Run the ex009_PHT3D_09 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
from examples.ex009_PHT3D_09.modflow_model import build_model
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
NROW = 31
NCOL = 51
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.3
PERLEN = 500.0
NSTP = 50
LENGTH_X = 510.0
LENGTH_Y = 310.0
TOP = 10.0
BOTM = 0.0
HYDRAULIC_CONDUCTIVITY = 50.0
ALH = 10.0
ATH1 = 3.0
ATV = 1.0
DIFFC = 0.0
INLET_HEAD = 100.0
OUTLET_HEAD = 99.0
INFLOW_RATE = 2.0
WELL_CELL = (0, 15, 15)


def main() -> None:
    ic_mapping = {"solution": 0, "kinetics": 1}
    simulation_config = SimulationConfig(
        case_name="ex009",
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
        fields=CellFields(porosity=POROSITY),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        build_model(
            workspace=WORKSPACE,
            species=species,
            initial_concentrations=initial_concentrations,
            background_concentrations=simulator.get_initial_concentrations(0),
            inflow_concentrations=simulator.get_initial_concentrations(1),
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
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            inflow_rate=INFLOW_RATE,
            well_cell=WELL_CELL,
            alh=ALH,
            ath1=ATH1,
            atv=ATV,
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        print("ex009 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
