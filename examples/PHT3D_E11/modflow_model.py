"""MODFLOW 6 flow and multicomponent transport model for PHT3D Example 11."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _example_support as _example_support
import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name

NLAY = 58
NROW = 1
NCOL = 99
NXYZ = NLAY * NROW * NCOL
FLOW_STEPS = 120
TRANSPORT_SUBSTEPS = 5
TOTAL_TRANSPORT_STEPS = FLOW_STEPS * TRANSPORT_SUBSTEPS

DELR = np.array(
    [1.0] * 7 + [0.5] * 2 + [0.25] * 76 + [0.5] * 2 + [1.0] * 12,
    dtype=float,
)
DELC = np.array([1.0], dtype=float)
TOP = 34.2
BOTM = np.array(
    [34.15 - 0.05 * layer for layer in range(44)]
    + [31.75, 31.5, 31.25, 31.0, 30.5, 30.0]
    + list(np.arange(29.0, 21.0, -1.0)),
    dtype=float,
)


def cell_centers_x() -> np.ndarray:
    """Return the official nonuniform-grid column centers in metres."""
    return np.cumsum(DELR) - 0.5 * DELR


def cell_centers_z() -> np.ndarray:
    """Return layer-center elevations in metres above sea level."""
    layer_tops = np.r_[TOP, BOTM[:-1]]
    return 0.5 * (layer_tops + BOTM)


def _split_initial_concentrations(
    species_list: list[str], initial_conc: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: initial_conc[index * NXYZ : (index + 1) * NXYZ]
        for index, name in enumerate(species_list)
    }


def transport_model(
    *,
    sim_ws: str,
    species_list: list[str],
    initial_conc: np.ndarray,
    ambient_concentrations: np.ndarray,
    recharge_concentrations: np.ndarray,
    mf6_exe: str,
    perlen: float = 60.0,
    nstp: int = 120,
) -> None:
    """Build the official E11 cross-section with one GWT model per component."""
    if DELR.size != NCOL or BOTM.size != NLAY:
        raise RuntimeError("E11 grid constants are inconsistent")

    sim_name = "model"
    gwf_name = f"gwf_{sim_name}"
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        sim_ws=sim_ws,
        exe_name=mf6_exe,
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(perlen, nstp, 1.0)],
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=gwf_name,
        save_flows=True,
        newtonoptions="NEWTON",
    )
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=1.0e-9,
        outer_maximum=200,
        inner_maximum=200,
        inner_dvclose=1.0e-10,
        rcloserecord=1.0e-8,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.97,
        filename="flow.ims",
    )
    sim.register_ims_package(flow_ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=DELC,
        top=TOP,
        botm=BOTM,
    )
    flopy.mf6.ModflowGwfic(
        gwf,
        pname="ic",
        strt=np.full((NLAY, NROW, NCOL), 33.76, dtype=float),
    )
    vertical_conductivity = np.full((NLAY, NROW, NCOL), 86.4, dtype=float)
    # The official BCF6 input uses 1000 m/d in the injection-well column so
    # the incoming water is distributed vertically over the screened layers.
    vertical_conductivity[:, 0, 0] = 1000.0
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=86.4,
        k33=vertical_conductivity,
        wetdry=-0.01,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        iconvert=1,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )

    # The original WEL package injects solute-free water along the saturated
    # part of the left boundary: every WELLS term in the PHT3D mass budgets is
    # zero. Ambient groundwater is imposed separately by the upstream CNC
    # boundary below. Rates are in m3/day for the 1 m-wide section.
    left_rates = np.r_[
        np.full(36, 2.67608e-2),
        np.full(4, 1.338042e-1),
        np.full(2, 2.676083e-1),
        np.full(8, 5.352167e-1),
    ]
    well_concentrations = np.zeros(len(species_list), dtype=float)
    left_wells = [
        [(layer, 0, 0), float(rate), *well_concentrations]
        for layer, rate in zip(range(8, NLAY), left_rates, strict=False)
    ]
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-LEFT",
        save_flows=True,
        maxbound=len(left_wells),
        stress_period_data={0: left_wells},
        auxiliary=species_list,
    )

    right_chd = [[(layer, 0, NCOL - 1), 33.76] for layer in range(8, NLAY)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-RIGHT",
        save_flows=True,
        maxbound=len(right_chd),
        stress_period_data={0: right_chd},
    )

    recharge = [[(0, 0, col), 1.0e-3, *recharge_concentrations] for col in range(NCOL)]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="RCH-TOP",
        fixed_cell=False,
        save_flows=True,
        maxbound=len(recharge),
        stress_period_data={0: recharge},
        auxiliary=species_list,
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    species_initial = _split_initial_concentrations(species_list, initial_conc)
    for species_name, concentration in species_initial.items():
        gwt_name = get_gwt_model_name(species_name)
        is_mobile = species_name != "Charge"
        # Tolu_h and Naph_h are roughly two orders of magnitude less
        # concentrated than their light partners.  A conventional absolute
        # residual produced 0.05% and 0.15% mass errors respectively, which
        # become visible 0.5--1.5 per-mil errors after taking isotope ratios.
        # Sulf_h is much more abundant and already mass-conservative at the
        # standard tolerances; forcing it to this low-concentration criterion
        # instead stalls the nonlinear solve in strongly depleted cells.
        needs_isotope_precision = species_name in {"Tolu_h", "Naph_h"}
        # TVD's nonlinear outer iterations cannot reliably reduce the
        # concentration-change norm below 1e-9 at the initial sharp fronts.
        # The isotope drift is instead controlled by the linear mass-rate
        # residual and inner increment, which must be far below the roughly
        # 1e-6 mol/day heavy-isotope fluxes in this case.
        transport_outer_dvclose = 1.0e-9 if needs_isotope_precision else 1.0e-7
        transport_inner_dvclose = 1.0e-12 if needs_isotope_precision else 1.0e-8
        transport_rclose = 1.0e-12 if needs_isotope_precision else 1.0e-7
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=False,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="NONE",
            complexity="MODERATE",
            outer_dvclose=transport_outer_dvclose,
            outer_maximum=200,
            inner_maximum=1000,
            inner_dvclose=transport_inner_dvclose,
            rcloserecord=transport_rclose,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            filename=f"{gwt_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            gwt,
            strt=concentration,
            filename=f"{gwt_name}.ic",
        )
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=0.30,
            filename=f"{gwt_name}.mst",
        )
        if is_mobile:
            flopy.mf6.ModflowGwtadv(
                gwt,
                scheme="TVD",
                filename=f"{gwt_name}.adv",
            )
            flopy.mf6.ModflowGwtdsp(
                gwt,
                alh=0.05,
                alv=0.05,
                ath1=0.005,
                # MT3D TRPV = 0.01 makes transverse spreading normal to
                # horizontal x-flow 0.05 * 0.01 = 0.0005 m in the z direction.
                ath2=0.0005,
                # For vertical flow, the transverse x/y value is AL * TRPT.
                atv=0.005,
                diffc=0.0,
                filename=f"{gwt_name}.dsp",
            )
            flopy.mf6.ModflowGwtssm(
                gwt,
                sources=[
                    ("WEL-LEFT", "AUX", species_name),
                    ("RCH-TOP", "AUX", species_name),
                ],
                filename=f"{gwt_name}.ssm",
            )

            # PHT3D fixes the first six columns to ambient concentrations.
            species_index = species_list.index(species_name)
            cnc_data = [
                [
                    (layer, 0, col),
                    float(ambient_concentrations[species_index]),
                ]
                for layer in range(NLAY)
                for col in range(6)
            ]
        else:
            # With CB_OFFSET=0 PHT3D resets charge imbalance before chemistry.
            # A full-grid zero CNC provides the same behaviour without adding
            # a component-specific branch to the reusable coupling module.
            cnc_data = [[(layer, 0, col), 0.0] for layer in range(NLAY) for col in range(NCOL)]
        flopy.mf6.ModflowGwtcnc(
            gwt,
            pname="CNC-UPSTREAM" if is_mobile else "CNC-ZERO-CHARGE",
            maxbound=len(cnc_data),
            stress_period_data={0: cnc_data},
            filename=f"{gwt_name}.cnc",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        if is_mobile:
            flopy.mf6.ModflowGwfgwt(
                sim,
                exgtype="GWF6-GWT6",
                exgmnamea=gwf_name,
                exgmnameb=gwt_name,
                filename=f"{gwt_name}.gwfgwt",
            )

    sim.write_simulation(silent=True)
