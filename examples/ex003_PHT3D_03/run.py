"""Run the ex003_PHT3D_03 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
from examples.ex003_PHT3D_03.modflow_model import build_model
from mf6pqc import MF6PQC, BackendPaths, CellFields, OutputOptions, SimulationConfig

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
NCOL = 80
NXYZ = NLAY * NROW * NCOL
DELR = 0.005
POROSITY = 0.35
ALH = 0.005
PERLEN = 24.0
NSTP = 192
INFLOW_RATE = 0.007
DELC = 1.0
TOP = 1.0
BOTM = 0.0
INITIAL_HEAD = 1.0
HYDRAULIC_CONDUCTIVITY = 0.056
DIFFC = 0.0
OUTLET_HEAD = 1.0


def main() -> None:
    simulation_config = SimulationConfig(
        case_name="ex003",
        nxyz=NXYZ,
        nthreads=6,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(pressure_atm=1.0, porosity=POROSITY),
        output=OutputOptions(save_steps=[NSTP // 4, NSTP // 2, NSTP]),
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "equilibrium_phases": 1})
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
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
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        print("ex003 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
