"""Run the 6000-year Hamann et al. (2015) playa RWM model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


CASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CASE_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mf6pqc.mf6pqc import mf6pqc
from modflow_model import (
    DAYS_PER_YEAR,
    build_reference_grid,
    build_time_config,
    build_transport_model,
    evaporation_rates_mm_per_year,
    water_only_sink_rates,
)


def main() -> None:
    grid = build_reference_grid()
    time_config = build_time_config()
    workspace = CASE_DIR / "simulation"
    output_dir = CASE_DIR / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    equilibrium_phases = np.full(grid.nxyz, 2, dtype=np.int32)
    equilibrium_phases[: grid.ncol] = 1
    simulator = mf6pqc(
        case_name="hamann2015",
        nxyz=grid.nxyz,
        nthreads=max(1, min(12, os.cpu_count() or 1)),
        temperature=25.0,
        pressure=2.0,
        porosity=0.25,
        saturation=1.0,
        density=0.99987,
        print_chemistry_mask=1,
        water_only_sink_rates=water_only_sink_rates(grid),
        # Hamann's evaporation boundary removes water while retaining salts.
        # H2O must therefore be transported explicitly.
        componentH2O=True,
        # Keep the transport control-volume fixed.  Calculated Pitzer density
        # is retrieved separately below and fed back to MODFLOW's BUY package.
        solution_density_volume=False,
        db_path=str(CASE_DIR / "input_data" / "pitzer.dat"),
        pqi_path=str(CASE_DIR / "input_data" / "input.pqi"),
        modflow_dll_path=str(REPO_ROOT / "bin" / "mf6.7.0" / "libmf6.dll"),
        workspace=str(workspace),
        output_dir=str(output_dir),
        if_update_porosity_K=False,
        if_update_density=True,
        use_phreeqc_calculated_density=True,
        save_interval=1,
        save_steps=list(time_config.snapshot_steps_global),
        progress_interval=100,
        fail_on_nonconvergence=True,
    )

    completed = False
    try:
        initial = simulator.setup(
            ic_map={
                "solution": 0,
                "equilibrium_phases": equilibrium_phases,
            }
        )
        recharge = simulator.get_initial_concentrations(0)
        components = simulator.get_components()
        build_transport_model(
            workspace=workspace,
            mf6_executable=REPO_ROOT / "bin" / "mf6.7.0" / "mf6.exe",
            grid=grid,
            time_config=time_config,
            species=components,
            initial_concentrations=initial,
            recharge_concentrations=recharge,
            porosity=0.25,
            hydraulic_conductivity_m_per_day=1.0e-5 * 86400.0,
        )
        metadata = {
            "paper": "Hamann et al. (2015), doi:10.1002/2015WR017833",
            "scenario": "RWM",
            "nlay": grid.nlay,
            "nrow": grid.nrow,
            "ncol": grid.ncol,
            "nxyz": grid.nxyz,
            "components": components,
            "componentH2O": True,
            "solution_density_volume": False,
            "pure_water_sink_solute_compensation": True,
            "snapshot_years": [0.0, *time_config.snapshot_years],
            "snapshot_steps_global": [0, *time_config.snapshot_steps_global],
            "perioddata_days": [list(record) for record in time_config.perioddata],
            "domain_length_m": float(grid.delr.sum()),
            "domain_depth_m": float(grid.delv.sum()),
            "max_delr_m": float(grid.delr.max()),
            "max_delv_m": float(grid.delv.max()),
            "evaporation_weighted_mean_mm_per_year": float(
                np.average(
                    evaporation_rates_mm_per_year(grid),
                    weights=grid.delr[grid.x_centres >= 50.0],
                )
            ),
        }
        (output_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        np.save(output_dir / "grid_delr_m.npy", grid.delr)
        np.save(output_dir / "grid_delv_m.npy", grid.delv)
        simulator.run()
        simulator.save_results()
        np.save(
            output_dir / "result_times_years.npy",
            np.asarray([0.0, *time_config.snapshot_years], dtype=float),
        )
        completed = True
    finally:
        simulator.finalize()
        status = "completed" if completed else "failed"
        print(f"Hamann 2015 run {status}.")


if __name__ == "__main__":
    main()
