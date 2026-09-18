"""Run the reactive column with coupled heat transport and viscosity feedback."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
from examples.ex999_Thermal_ReactiveColumn1D.modflow_model import build_model
from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    EnergyOptions,
    FeedbackOptions,
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
NCOL = 41
NXYZ = NLAY * NROW * NCOL
DELR = 0.25
POROSITY = 0.3
HYDRAULIC_CONDUCTIVITY = 1.0
INITIAL_TEMPERATURE = 20.0
INFLOW_TEMPERATURE = 60.0
PERLEN = 20.0
NSTP = 40
VISCOSITY_REFERENCE = 0.0010016
DELC = 1.0
TOP = 1.0
BOTM = 0.0
DENSITY_WATER = 1000.0
HEAT_CAPACITY_WATER = 4184.0
DENSITY_SOLID = 2650.0
HEAT_CAPACITY_SOLID = 800.0
ALH = 0.05
ATH1 = 0.005
ATV = 0.005
KTW = 0.6 * 86400.0
KTS = 2.0 * 86400.0
DIFFC = 1e-05
INLET_HEAD = 1.0
OUTLET_HEAD = 0.0
THERMAL_A2 = 10.0
THERMAL_A3 = 248.37
THERMAL_A4 = 133.15


def main() -> None:
    simulation_config = SimulationConfig(
        case_name="ex999",
        nxyz=NXYZ,
        nthreads=4,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=WORKSPACE,
            output_directory=OUTPUT_DIR,
        ),
        fields=CellFields(temperature_c=INITIAL_TEMPERATURE, porosity=POROSITY),
        feedback=FeedbackOptions(
            update_porosity_and_k=True, mineral_molar_volumes={"ThermalMineral": 0.04}
        ),
        energy=EnergyOptions(
            enabled=True,
            viscosity_feedback=True,
            flow_model_name="gwf_model",
            energy_model_name="gwe_model",
            sync_temperature_to_chemistry=True,
            validate_initial_fields=True,
        ),
        output=OutputOptions(progress_interval=10),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "kinetics": 1})
        inflow_concentrations = simulator.get_initial_concentrations(1)
        species = simulator.get_components()
        if "Charge" in species:
            charge_index = species.index("Charge")
            start = charge_index * NXYZ
            initial_concentrations[start : start + NXYZ] = 0.0
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
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            atv=ATV,
            diffc=DIFFC,
            density_solid=DENSITY_SOLID,
            density_water=DENSITY_WATER,
            heat_capacity_solid=HEAT_CAPACITY_SOLID,
            heat_capacity_water=HEAT_CAPACITY_WATER,
            inflow_temperature=INFLOW_TEMPERATURE,
            initial_temperature=INITIAL_TEMPERATURE,
            kts=KTS,
            ktw=KTW,
            thermal_a2=THERMAL_A2,
            thermal_a3=THERMAL_A3,
            thermal_a4=THERMAL_A4,
            viscosity_reference=VISCOSITY_REFERENCE,
        )
        simulator.run(method="ThermalSNIA")
        simulator.save_results()
        print("ex999 done.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
