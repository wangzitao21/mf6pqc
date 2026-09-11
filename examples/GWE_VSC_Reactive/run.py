"""Run the integrated thermal, viscous, reactive-transport example."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os

from modflow_model import (
    INITIAL_TEMPERATURE,
    NXYZ,
    POROSITY,
    build_model,
    configure_logging,
    executable_path,
    library_path,
    runtime_path,
)

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


def main() -> None:
    EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

    config = SimulationConfig(
        case_name="GWE_VSC_Reactive",
        nxyz=NXYZ,
        nthreads=4,
        paths=BackendPaths(
            database=os.path.join(EXAMPLE_DIR, "input_data", "database.dat"),
            chemistry_input=os.path.join(EXAMPLE_DIR, "input_data", "input.pqi"),
            modflow_library=library_path("mf6.8.0"),
            workspace=runtime_path(__file__, "simulation"),
            output_directory=runtime_path(__file__, "output"),
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
            update_porosity_and_k=True,
            mineral_molar_volumes={"ThermalMineral": 0.040},
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

    with MF6PQC.from_config(config) as simulator:
        initial = simulator.setup({"solution": 0, "kinetics": 1})
        inflow = simulator.get_initial_concentrations(1)
        # Charge is a PHREEQC numerical residual rather than a transported mass.
        # MODFLOW specified-concentration packages require a nonnegative value.
        if "Charge" in simulator.get_components():
            charge_index = simulator.get_components().index("Charge")
            start = charge_index * NXYZ
            initial[start : start + NXYZ] = 0.0
            inflow[charge_index] = 0.0
        build_model(
            sim_ws=str(config.paths.workspace),
            species_list=simulator.get_components(),
            initial_concentrations=initial,
            inflow_concentrations=inflow,
            mf6_exe=executable_path("mf6.8.0"),
        )
        simulator.run(method="ThermalSNIA")
        simulator.save_results()

        print("GWE_VSC_Reactive completed. Open plot.ipynb for quantitative checks.")


if __name__ == "__main__":
    configure_logging()
    main()
