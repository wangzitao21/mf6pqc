"""Run the ex013_PHT3D_13 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex013_PHT3D_13.modflow_model import build_model
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
NCOL = 16
NXYZ = NLAY * NROW * NCOL
DELR = 0.0033125
DELC = 1.0
TOP = 0.00287433
BOTM = 0.0
POROSITY = 0.376
INFLOW_RATE = 0.00024
FLOW_PERIOD_DATA = [(0.9333333, 64, 1.0), (1.458333, 100, 1.0)]
TRANSPORT_SUBSTEPS = 4
PERIOD_DATA = [
    (length, steps * TRANSPORT_SUBSTEPS, multiplier)
    for length, steps, multiplier in FLOW_PERIOD_DATA
]
INITIAL_HEAD = 1.0
HYDRAULIC_CONDUCTIVITY = 1.0
ALH = 0.00537
ATH1 = 0.000537
DIFFC = 0.0
OUTLET_HEAD = 1.0


def main() -> None:
    zone_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    ic_mapping = {
        "solution": 0,
        "equilibrium_phases": zone_ids,
        "exchange": zone_ids,
        "surface": zone_ids,
        "kinetics": zone_ids,
    }
    simulation_config = SimulationConfig(
        case_name="ex013",
        nxyz=NXYZ,
        nthreads=6,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(temperature_c=7.0, pressure_atm=1.0, porosity=POROSITY),
        output=OutputOptions(progress_interval=10),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        pulse_concentrations = simulator.get_initial_concentrations(1)
        chase_concentrations = simulator.get_initial_concentrations(2)
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
        print(f"Saved {NCOL}-cell outlet histories for both stress periods.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
