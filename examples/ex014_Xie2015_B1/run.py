"""Run the ex014_Xie2015_B1 reactive-transport benchmark."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex014_Xie2015_B1.modflow_model import build_model
from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    FeedbackOptions,
    ImplicitOptions,
    KineticReaction,
    OutputOptions,
    SimulationConfig,
)

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
NCOL = 81
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.35
DELR = [0.0125] + [0.025] * (NCOL - 2) + [0.0125]
DELC = [1.0]
TOP = 1.0
BOTM = 0
ALH = 0.0
ATH1 = ALH / 10
DIFFC = 0.0
BOUNDARY_DISTANCE = DELR[0] / 2
INLET_HEAD = 0.007
OUTLET_HEAD = 0.0
VERTICAL_CONDUCTIVITY_RATIO = 0.1
HYDRAULIC_CONDUCTIVITY = 0.000116 * 86400.0
INITIAL_HEAD = 0.0
CALCITE_MOLAR_VOLUME = 100.0894 / 2710.0
INITIAL_CALCITE_VOLUME_FRACTION = 0.3
DEFAULT_MAX_STEP_YEARS = 0.3


def time_discretization(years=500.0, max_step_years=DEFAULT_MAX_STEP_YEARS):
    if (
        not np.isfinite(years)
        or years <= 0
        or (not np.isfinite(max_step_years))
        or (max_step_years <= 0)
    ):
        raise ValueError("Simulation duration and maximum time step must be positive and finite")
    ends = sorted({y for y in (1, 10, 100, 120, 130, 150, 200, 300, 500, years) if y <= years})
    periods, time_steps = ([], [])
    previous = 0.0
    for end in ends:
        step = max_step_years
        count, multiplier = (int(np.ceil((end - previous) / step)), 1.0)
        if previous == 0 and end == 1:
            count, multiplier = (60, 1.15)
            while (1 - 1 / multiplier) / (1 - multiplier ** (-count)) > step:
                multiplier = np.sqrt(multiplier)
                count *= 2
        duration = (end - previous) * 365.0
        weights = multiplier ** (np.arange(count, dtype=float) - count + 1)
        durations = duration * weights / weights.sum()
        durations[-1] += duration - durations.sum()
        periods.append((duration, count, multiplier))
        time_steps.extend(durations)
        previous = end
    times = np.cumsum(time_steps)
    targets = sorted({*ends, *np.arange(1, min(years, 150) + 1), *np.arange(160, years + 1, 10)})
    save_steps = sorted({int(np.abs(times - y * 365).argmin()) + 1 for y in targets})
    return (periods, save_steps)


def main() -> None:
    periods, save_steps = time_discretization(500.0, DEFAULT_MAX_STEP_YEARS)
    ic_mapping = {"solution": 0, "kinetics": 1}
    simulation_config = SimulationConfig(
        case_name="ex014",
        nxyz=NXYZ,
        nthreads=4,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(pressure_atm=1.0, porosity=POROSITY),
        feedback=FeedbackOptions(
            update_porosity_and_k=True,
            mineral_molar_volumes={"Calcite": CALCITE_MOLAR_VOLUME},
            vertical_to_horizontal_k_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            fail_on_porosity_clipping=True,
            boundary_conductance_updates={
                "BUSHUI": {"cell_index": 0, "distance": BOUNDARY_DISTANCE},
                "GHB_RIGHT": {"cell_index": -1, "distance": BOUNDARY_DISTANCE},
            },
        ),
        output=OutputOptions(save_steps=save_steps),
        fail_on_modflow_nonconvergence=True,
    )
    parameters = re.findall(
        "(?im)^\\s*-parms\\s+([\\d.eE+\\-]+)\\s*$", (INPUT_DIR / "input.pqi").read_text()
    )
    if len(parameters) != 1:
        raise ValueError("B1 input must define one Calcite rate constant")
    simulation_config.implicit = ImplicitOptions(
        reactions=(
            KineticReaction(
                "Calcite",
                {"Ca": 1, "C": 1, "O": 3},
                float(parameters[0]) * 86400.0,
                surface_exponent=2 / 3,
            ),
        )
    )
    hydraulic_conductivity = np.full((NLAY, NROW, NCOL), HYDRAULIC_CONDUCTIVITY)
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
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
            period_data=periods,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            vertical_conductivity_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            initial_head=INITIAL_HEAD,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            boundary_distance=BOUNDARY_DISTANCE,
        )
        simulator.run("Implicit")
        simulator.save_results()
        print("ex014 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
