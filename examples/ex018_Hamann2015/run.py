"""Run the ex018_Hamann2015 reactive-transport benchmark."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex018_Hamann2015.modflow_model import Grid, TimeConfig, build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    FeedbackOptions,
    OutputOptions,
    SimulationConfig,
)

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
DAYS_PER_YEAR = 365.25
POROSITY = 0.25
HYDRAULIC_CONDUCTIVITY = 1e-05 * 86400.0
RECHARGE_RATE = 80.0 / 1000.0 / DAYS_PER_YEAR
ALH = 2.0
ATH1 = 1.0
DIFFC = 1e-09 * 86400.0
REFERENCE_DENSITY = 1000.0
SPLIT_X = 50.0


def build_reference_grid() -> Grid:
    recharge_delr = np.ones(50, dtype=float)
    playa_raw = np.geomspace(1.0, 0.1, 128)
    playa_delr = playa_raw * (50.0 / playa_raw.sum())
    delr = np.r_[recharge_delr, playa_delr]
    surface_layer = 0.062
    deeper_layers = np.full(10, (10.0 - surface_layer) / 10.0)
    delv = np.r_[surface_layer, deeper_layers]
    grid = Grid(delr=delr, delv=delv, top=10.0)
    if not np.isclose(grid.delr.sum(), 100.0):
        raise AssertionError("Grid length must be 100 m")
    if not np.isclose(grid.delv.sum(), 10.0):
        raise AssertionError("Grid depth must be 10 m")
    if grid.delr.max() > 1.0 + 1e-12 or grid.delv.max() > 1.0 + 1e-12:
        raise AssertionError("Reference grid must satisfy the <=1 m paper criterion")
    return grid


def build_time_config() -> TimeConfig:
    perioddata = ((1000.0 * DAYS_PER_YEAR, 10000, 1.0), (5000.0 * DAYS_PER_YEAR, 50000, 1.0))
    snapshots = (1.0, 20.0, 40.0, 70.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
    global_steps: list[int] = []
    by_period: dict[int, list[int]] = {}
    for year in snapshots:
        found = False
        period_start_year = 0.0
        cumulative_steps = 0
        for kper, (perlen_days, nstp, _) in enumerate(perioddata):
            period_years = perlen_days / DAYS_PER_YEAR
            if period_start_year < year <= period_start_year + period_years + 1e-12:
                dt_years = period_years / nstp
                local_step = int(round((year - period_start_year) / dt_years))
                if not np.isclose(period_start_year + local_step * dt_years, year):
                    raise ValueError(f"Snapshot year {year} is not on a time-step boundary")
                global_steps.append(cumulative_steps + local_step)
                by_period.setdefault(kper, []).append(local_step)
                found = True
                break
            period_start_year += period_years
            cumulative_steps += nstp
        if not found:
            raise ValueError(f"Snapshot year {year} exceeds the simulation duration")
    return TimeConfig(
        perioddata=perioddata,
        snapshot_years=snapshots,
        snapshot_steps_global=tuple(global_steps),
        snapshot_steps_by_period=by_period,
    )


def evaporation_rates_mm_per_year(grid: Grid, split_x: float = SPLIT_X) -> np.ndarray:
    x = grid.x_centres
    mask = x >= split_x
    rates = 92.0 - 24.0 * ((x[mask] - split_x) / (100.0 - split_x))
    weighted_mean = np.average(rates, weights=grid.delr[mask])
    if not np.isclose(weighted_mean, 80.0, atol=1e-12):
        raise AssertionError(f"Evaporation water balance failed: mean={weighted_mean}")
    return rates


def water_only_sink_rates(grid: Grid, split_x: float = SPLIT_X) -> np.ndarray:
    rates = np.zeros(grid.nxyz, dtype=float)
    split_col = int(np.count_nonzero(grid.x_centres < split_x))
    evaporation = evaporation_rates_mm_per_year(grid, split_x=split_x)
    rates[split_col : grid.ncol] = evaporation / 1000.0 / DAYS_PER_YEAR * grid.delr[split_col:]
    return rates


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    grid = build_reference_grid()
    evaporation_rates = evaporation_rates_mm_per_year(grid, split_x=SPLIT_X)
    time_config = build_time_config()
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    output_dir.mkdir(parents=True, exist_ok=True)
    equilibrium_phases = np.full(grid.nxyz, 2, dtype=np.int32)
    equilibrium_phases[: grid.ncol] = 1
    simulation_config = SimulationConfig(
        case_name="ex018",
        nxyz=grid.nxyz,
        nthreads=max(1, min(12, os.cpu_count() or 1)),
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=25.0,
            pressure_atm=2.0,
            porosity=POROSITY,
            saturation=1.0,
            density_kg_per_litre=0.99987,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=True,
            use_solution_density_volume=False,
        ),
        feedback=FeedbackOptions(
            update_porosity_and_k=False,
            update_density=True,
            water_only_sink_rates=water_only_sink_rates(grid, split_x=SPLIT_X),
            use_phreeqc_calculated_density=True,
        ),
        output=OutputOptions(
            save_interval=1,
            save_steps=list(time_config.snapshot_steps_global),
            progress_interval=100,
        ),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(
            ic_map={"solution": 0, "equilibrium_phases": equilibrium_phases}
        )
        species = simulator.get_components()
        recharge_concentrations = simulator.get_initial_concentrations(0)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            recharge_concentrations=recharge_concentrations,
            grid=grid,
            time_config=time_config,
            porosity=POROSITY,
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            recharge_rate=RECHARGE_RATE,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            days_per_year=DAYS_PER_YEAR,
            evaporation_rates=evaporation_rates,
            reference_density=REFERENCE_DENSITY,
            split_x=SPLIT_X,
        )
        metadata = {
            "paper": "Hamann et al. (2015), doi:10.1002/2015WR017833",
            "scenario": "RWM",
            "nlay": grid.nlay,
            "nrow": grid.nrow,
            "ncol": grid.ncol,
            "nxyz": grid.nxyz,
            "components": species,
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
                np.average(evaporation_rates, weights=grid.delr[grid.x_centres >= SPLIT_X])
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
    print("ex018 done.")


if __name__ == "__main__":
    configure_logging()
    main()
