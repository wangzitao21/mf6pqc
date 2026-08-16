"""MODFLOW 6 flow and multicomponent transport model for Hamann et al. (2015).

The model represents the reactive-with-minerals (RWM) scenario.  All lengths
are in metres, time is in days, hydraulic conductivity is in metres per day,
and transported component concentrations are in mol/L.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import flopy
import numpy as np


DAYS_PER_YEAR = 365.25


@dataclass(frozen=True)
class Grid:
    """Structured cross-section and derived cell-centre coordinates."""

    delr: np.ndarray
    delv: np.ndarray
    top: float

    @property
    def nlay(self) -> int:
        return int(self.delv.size)

    @property
    def nrow(self) -> int:
        return 1

    @property
    def ncol(self) -> int:
        return int(self.delr.size)

    @property
    def nxyz(self) -> int:
        return self.nlay * self.nrow * self.ncol

    @property
    def botm(self) -> np.ndarray:
        return self.top - np.cumsum(self.delv)

    @property
    def x_centres(self) -> np.ndarray:
        return np.cumsum(self.delr) - 0.5 * self.delr

    @property
    def z_centres(self) -> np.ndarray:
        layer_tops = np.r_[self.top, self.botm[:-1]]
        return layer_tops - 0.5 * self.delv


@dataclass(frozen=True)
class TimeConfig:
    """TDIS definition and exact output steps for the 6000-year run."""

    perioddata: tuple[tuple[float, int, float], ...]
    snapshot_years: tuple[float, ...]
    snapshot_steps_global: tuple[int, ...]
    snapshot_steps_by_period: dict[int, list[int]]


def build_reference_grid() -> Grid:
    """Return a grid satisfying the paper's <=1 m resolution criteria.

    The first layer is 0.062 m thick, matching the grid-converged evaporation
    boundary layer reported by Hamann et al.  The playa half is gradually
    refined from about 1 m at x=50 m to about 0.1 m at x=100 m.
    """

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
    if grid.delr.max() > 1.0 + 1.0e-12 or grid.delv.max() > 1.0 + 1.0e-12:
        raise AssertionError("Reference grid must satisfy the <=1 m paper criterion")
    return grid


def build_time_config() -> TimeConfig:
    """Return the 6000-year discretisation used by this MF6PQC case.

    Hamann et al. used 0.1-year flow steps for the first 1000 years and
    1-year flow steps thereafter, but their transport solver introduced
    smaller internal steps as required by the Courant number.  MF6PQC
    currently advances transport and chemistry once per MODFLOW time step,
    so this case retains a 0.1-year step for the entire simulation to avoid
    the severe late-time splitting error produced by 1-year transport steps.
    """

    perioddata = (
        (1000.0 * DAYS_PER_YEAR, 10_000, 1.0),
        (5000.0 * DAYS_PER_YEAR, 50_000, 1.0),
    )
    snapshots = (
        1.0,
        20.0,
        40.0,
        70.0,
        1000.0,
        2000.0,
        3000.0,
        4000.0,
        5000.0,
        6000.0,
    )

    global_steps: list[int] = []
    by_period: dict[int, list[int]] = {}
    for year in snapshots:
        found = False
        period_start_year = 0.0
        cumulative_steps = 0
        for kper, (perlen_days, nstp, _) in enumerate(perioddata):
            period_years = perlen_days / DAYS_PER_YEAR
            if period_start_year < year <= period_start_year + period_years + 1.0e-12:
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


def evaporation_rates_mm_per_year(grid: Grid, split_x: float = 50.0) -> np.ndarray:
    """Return the paper's 92-to-68 mm/y linear evaporation profile.

    Rates are evaluated at physical cell centres, rather than interpolated by
    cell number.  Midpoint integration of this linear profile gives exactly
    the prescribed 80 mm/y spatial mean on the nonuniform playa grid.
    """

    x = grid.x_centres
    mask = x >= split_x
    rates = 92.0 - 24.0 * ((x[mask] - split_x) / (100.0 - split_x))
    weighted_mean = np.average(rates, weights=grid.delr[mask])
    if not np.isclose(weighted_mean, 80.0, atol=1.0e-12):
        raise AssertionError(f"Evaporation water balance failed: mean={weighted_mean}")
    return rates


def _output_records(time_config: TimeConfig, variable: str) -> dict[int, list[tuple]]:
    return {
        kper: [(variable, "STEPS", *steps)]
        for kper, steps in time_config.snapshot_steps_by_period.items()
    }


def build_transport_model(
    *,
    workspace: Path,
    mf6_executable: Path,
    grid: Grid,
    time_config: TimeConfig,
    species: Iterable[str],
    initial_concentrations: np.ndarray,
    recharge_concentrations: np.ndarray,
    porosity: float = 0.25,
    hydraulic_conductivity_m_per_day: float = 0.864,
) -> flopy.mf6.MFSimulation:
    """Create and write the coupled GWF/GWT simulation input files."""

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    species = list(species)
    initial_concentrations = np.asarray(initial_concentrations, dtype=float)
    recharge_concentrations = np.asarray(recharge_concentrations, dtype=float)
    if initial_concentrations.size != len(species) * grid.nxyz:
        raise ValueError("Initial concentration vector does not match species x cells")
    if recharge_concentrations.size != len(species):
        raise ValueError("Recharge concentration vector does not match species count")

    sim = flopy.mf6.MFSimulation(
        sim_name="hamann2015",
        sim_ws=str(workspace),
        exe_name=str(mf6_executable),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=len(time_config.perioddata),
        perioddata=list(time_config.perioddata),
    )

    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwf_name,
        save_flows=True,
    )
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=1.0e-4,
        outer_maximum=100,
        inner_maximum=500,
        inner_dvclose=1.0e-6,
        rcloserecord=1.0e-6,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.97,
        filename="flow.ims",
    )
    sim.register_ims_package(flow_ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="DIS",
        nlay=grid.nlay,
        nrow=grid.nrow,
        ncol=grid.ncol,
        delr=grid.delr,
        delc=1.0,
        top=grid.top,
        botm=grid.botm,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity_m_per_day,
        k33=hydraulic_conductivity_m_per_day,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="IC", strt=grid.top)
    flopy.mf6.ModflowGwfsto(
        gwf,
        pname="STO",
        save_flows=True,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        transient={0: True},
    )

    split_col = int(np.count_nonzero(grid.x_centres < 50.0))
    recharge_rate = (80.0 / 1000.0) / DAYS_PER_YEAR
    recharge_spd = [
        ((0, 0, j), recharge_rate, *recharge_concentrations)
        for j in range(split_col)
    ]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="RECHARGE",
        filename="gwf_model_recharge.rch",
        fixed_cell=True,
        auxiliary=species,
        stress_period_data={0: recharge_spd},
    )

    evaporation_rates = evaporation_rates_mm_per_year(grid)
    # GWT removes mass at the cell concentration for a negative water flux.
    # Dynamic SRC terms cancel this removal for solutes; H2O remains
    # uncompensated so the boundary represents evaporation of pure water.
    evaporation_aux = np.zeros(len(species), dtype=float)
    evaporation_spd = [
        (
            (0, 0, j),
            -(rate / 1000.0) / DAYS_PER_YEAR,
            *evaporation_aux,
        )
        for j, rate in zip(range(split_col, grid.ncol), evaporation_rates)
    ]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="EVAPORATION",
        filename="gwf_model_evaporation.rch",
        fixed_cell=True,
        auxiliary=species,
        stress_period_data={0: evaporation_spd},
    )

    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="HEAD_REFERENCE",
        auxiliary=species,
        stress_period_data={0: [((0, 0, 0), grid.top, *recharge_concentrations)]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="OC",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord={
            kper: [
                ("HEAD", "STEPS", *steps),
                ("BUDGET", "STEPS", *steps),
            ]
            for kper, steps in time_config.snapshot_steps_by_period.items()
        },
    )

    species_initial = {}
    for i, name in enumerate(species):
        start = i * grid.nxyz
        species_initial[name] = initial_concentrations[start : start + grid.nxyz]

    for name, initial in species_initial.items():
        gwt_name = f"gwt_{name}_model"
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            model_nam_file=f"{gwt_name}.nam",
            save_flows=False,
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=1.0e-6,
            outer_maximum=100,
            inner_maximum=200,
            inner_dvclose=1.0e-7,
            rcloserecord=1.0e-6,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=grid.nlay,
            nrow=grid.nrow,
            ncol=grid.ncol,
            delr=grid.delr,
            delc=1.0,
            top=grid.top,
            botm=grid.botm,
            filename=f"{gwt_name}.dis",
        )
        flopy.mf6.ModflowGwtic(gwt, strt=initial, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=2.0,
            alv=2.0,
            ath1=1.0,
            atv=1.0,
            diffc=1.0e-9 * 86400.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        flopy.mf6.ModflowGwtsrc(
            gwt,
            pname="SRC",
            maxbound=grid.nxyz,
            stress_period_data={
                0: [
                    ((k, 0, j), 0.0)
                    for k in range(grid.nlay)
                    for j in range(grid.ncol)
                ]
            },
            filename=f"{gwt_name}.src",
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            pname=f"{name}_SSM",
            sources=[
                ("HEAD_REFERENCE", "AUX", name),
                ("RECHARGE", "AUX", name),
                ("EVAPORATION", "AUX", name),
            ],
            filename=f"{gwt_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=_output_records(time_config, "CONCENTRATION"),
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )

    # A zero-slope BUY species creates the density array that MF6PQC updates
    # directly from PHREEQC's RHO result before each flow solve.
    flopy.mf6.ModflowGwfbuy(
        gwf,
        pname="BUY",
        denseref=1000.0,
        nrhospecies=1,
        density_filerecord="model_density.bin",
        packagedata=[(0, 0.0, 0.0, "gwt_Cl_model", "CONCENTRATION")],
    )

    sim.write_simulation(silent=False)
    return sim


def water_only_sink_rates(grid: Grid) -> np.ndarray:
    """Return positive evaporation volumes (m3/day) for every model cell."""
    rates = np.zeros(grid.nxyz, dtype=float)
    split_col = int(np.count_nonzero(grid.x_centres < 50.0))
    evaporation = evaporation_rates_mm_per_year(grid)
    rates[split_col : grid.ncol] = (
        (evaporation / 1000.0)
        / DAYS_PER_YEAR
        * grid.delr[split_col:]
    )
    return rates
