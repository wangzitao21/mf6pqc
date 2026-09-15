"""Run the ex001_PHT3D_01 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex001_PHT3D_01.modflow_model import build_model
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
NCOL = 150
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.25
PERLEN = 1826.0
NSTP = 200
DELR = 1.0
DELC = 1.0
TOP = 1.0
BOTM = 0.0
INITIAL_HEAD = 1.0
HYDRAULIC_CONDUCTIVITY = 1.0
ALH = 0.0
ATH1 = 0.0
ATV = 0.0
DIFFC = 0.0
OUTLET_HEAD = 1.0
INLET_HEAD = 4.725


def main() -> None:
    solution_map = np.zeros(NXYZ, dtype=np.int32)
    solution_map[0] = 1
    kinetics_map = np.ones(NXYZ, dtype=np.int32)
    kinetics_map[0] = -1
    ic_mapping = {"solution": solution_map, "kinetics": kinetics_map}
    simulation_config = SimulationConfig(
        case_name="ex001",
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
        output=OutputOptions(save_steps=[NSTP], progress_interval=20),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
        charge_index = species.index("Charge")
        initial_concentrations[charge_index * NXYZ : (charge_index + 1) * NXYZ] = 0.0
        inflow_concentrations[charge_index] = 0.0
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
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            atv=ATV,
            diffc=DIFFC,
        )
        simulator.run()
        simulator.save_results()
        print("ex001 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
