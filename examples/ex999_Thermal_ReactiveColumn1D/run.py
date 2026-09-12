"""Run the reactive column with coupled heat transport and viscosity feedback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from ex999_Thermal_ReactiveColumn1D.modflow_model import build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    EnergyOptions,
    FeedbackOptions,
    OutputOptions,
    SimulationConfig,
)

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
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
    """Configure, build, run, and save the thermal reactive-transport example."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    simulation_config = SimulationConfig(
        case_name="ex999",
        nxyz=NXYZ,
        nthreads=4,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=INITIAL_TEMPERATURE,
            pressure_atm=2.0,
            porosity=POROSITY,
            saturation=1.0,
            density_kg_per_litre=1.0,
        ),
        chemistry=ChemistryOptions(print_chemistry_mask=0),
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
        output=OutputOptions(save_interval=1, progress_interval=10),
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
            workspace=workspace,
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
    configure_logging()
    main()
