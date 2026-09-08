"""Run the MF6PQC reproduction of PHT3D Example 3."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _example_support as _example_support
from _example_support import library_path, runtime_path
from modflow_model import NCOL, POROSITY, TIME_STEPS, transport_model  # noqa: E402

from mf6pqc import MF6PQC  # noqa: E402


def main() -> None:
    CASE_DIR = Path(__file__).resolve().parent
    REPOSITORY_DIR = CASE_DIR.parents[1]

    with MF6PQC(
        case_name="PHT3D_E03",
        nxyz=NCOL,
        nthreads=6,
        temperature=25.0,
        pressure=1.0,
        porosity=POROSITY,
        saturation=1.0,
        density=1.0,
        print_chemistry_mask=0,
        componentH2O=False,
        solution_density_volume=False,
        db_path=str(CASE_DIR / "input_data" / "phreeqc.dat"),
        pqi_path=str(CASE_DIR / "input_data" / "phreeqc.pqi"),
        modflow_dll_path=library_path("mf6.8.0"),
        workspace=str(runtime_path(__file__, "simulation")),
        output_dir=str(runtime_path(__file__, "output")),
        if_update_porosity_K=False,
        if_update_density=False,
        save_steps=[TIME_STEPS // 4, TIME_STEPS // 2, TIME_STEPS],
    ) as simulator:
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "equilibrium_phases": 1})
        inflow_concentrations = simulator.get_initial_concentrations(1)

        transport_model(
            sim_ws=runtime_path(__file__, "simulation"),
            species_list=simulator.get_components(),
            initial_conc=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            mf6_exe=REPOSITORY_DIR / "bin" / "mf6.8.0" / "mf6.exe",
        )

        simulator.run()
        simulator.save_results()

        print("\n-------------------------------------------")
        print("PHT3D_E03 done")
        print("-------------------------------------------\n")


if __name__ == "__main__":
    _example_support.configure_logging()
    main()
