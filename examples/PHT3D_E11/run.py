from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os

import numpy as np
from modflow_model import (
    NCOL,
    NLAY,
    NROW,
    NXYZ,
    TOTAL_TRANSPORT_STEPS,
    TRANSPORT_SUBSTEPS,
    configure_logging,
    executable_path,
    library_path,
    runtime_path,
    transport_model,
)

from mf6pqc import MF6PQC


def main() -> None:
    EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

    POROSITY = 0.30
    PHT3D_REACTION_WATER_VOLUME_L = 1.0

    # The official model has 120 half-day flow/reaction steps.  Its MMOC
    # advection calculation takes five internal transport substeps per flow step,
    # but PHREEQC is called only at the end of the half-day step (OS=2).  Retain
    # the smaller transport interval for the MF6 discretisation without turning
    # those internal substeps into extra reaction calls.
    REACTION_STEPS = list(
        range(
            TRANSPORT_SUBSTEPS,
            TOTAL_TRANSPORT_STEPS + 1,
            TRANSPORT_SUBSTEPS,
        )
    )
    SAVE_STEPS = list(range(50, TOTAL_TRANSPORT_STEPS + 1, 50))

    # PHT3D uses two NAPL sources beginning at column 14 (one-based). The
    # generated phinp.dat expands each explicit source cell with COPY KINETICS:
    # the upper source spans columns 14-21 and layers 9-14, while the lower source
    # spans columns 14-17 in layer 24.
    kinetics_map = np.zeros((NLAY, NROW, NCOL), dtype=np.int32)
    kinetics_map[8:14, 0, 13:21] = 1
    kinetics_map[23, 0, 13:17] = 1

    ic_mapping = {
        "solution": 0,
        "equilibrium_phases": 1,
        "kinetics": kinetics_map.ravel(),
    }

    input_data_dir = os.path.join(EXAMPLE_DIR, "input_data")
    sim_params = {
        "case_name": "PHT3D_E11",
        "nxyz": NXYZ,
        "nthreads": 6,
        "temperature": 15.0,
        "pressure": 2.0,
        "porosity": POROSITY,
        "saturation": 1.0,
        "density": 1.0,
        "print_chemistry_mask": 0,
        "componentH2O": False,
        # PHT3D's CB_OFFSET=0 discards charge imbalance instead of transporting
        # it.  The case-local Charge GWT model below therefore holds it at zero.
        "signed_components": (),
        "solution_density_volume": False,
        "db_path": os.path.join(input_data_dir, "phreeqc.dat"),
        "pqi_path": os.path.join(input_data_dir, "input.pqi"),
        "modflow_dll_path": library_path("mf6.8.0"),
        "workspace": runtime_path(__file__, "simulation"),
        "output_dir": runtime_path(__file__, "output"),
        "if_update_porosity_K": False,
        "if_update_density": False,
        "save_steps": SAVE_STEPS,
        "reaction_steps": REACTION_STEPS,
        "progress_interval": 50,
        "fail_on_nonconvergence": True,
    }

    with MF6PQC(**sim_params) as simulator:
        # PHT3D runs each grid-cell chemistry calculation with one litre of water.
        # PhreeqcRM instead defaults to a one-litre representative (bulk) volume, so
        # at porosity 0.30 its default reaction water volume would be only 0.30 L.
        # Use water-based units and a 1 / porosity representative volume to retain the
        # official PHT3D amounts and kinetic rates without calibrating rate constants.
        simulator.phreeqc_rm.SetUnitsKinetics(1)
        simulator.phreeqc_rm.SetUnitsPPassemblage(1)
        simulator.phreeqc_rm.SetRepresentativeVolume(
            np.full(NXYZ, PHT3D_REACTION_WATER_VOLUME_L / POROSITY, dtype=float)
        )

        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        components = simulator.get_components()
        ambient_concentrations = simulator.get_initial_concentrations(0)
        recharge_concentrations = simulator.get_initial_concentrations(1)

        # PHREEQC's transported Charge component is a numerical residual and can be
        # slightly negative (about 4e-7 mol/L here). MODFLOW 6 correctly rejects a
        # negative specified concentration, while PhreeqcRM reconstructs charge
        # balance after every transport step. Use zero for that residual in MF6.
        charge_index = components.index("Charge")
        initial_concentrations[charge_index * NXYZ : (charge_index + 1) * NXYZ] = 0.0
        ambient_concentrations[charge_index] = 0.0
        recharge_concentrations[charge_index] = 0.0

        transport_model(
            sim_ws=sim_params["workspace"],
            species_list=components,
            initial_conc=initial_concentrations,
            ambient_concentrations=ambient_concentrations,
            recharge_concentrations=recharge_concentrations,
            mf6_exe=executable_path("mf6.8.0"),
            nstp=TOTAL_TRANSPORT_STEPS,
        )

        simulator.run()
        simulator.save_results()

        print("\n-------------------------------------------")
        print(f"'{sim_params['case_name']}' done.")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    configure_logging()
    main()
