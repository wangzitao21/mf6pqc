"""MODFLOW 6 flow and multicomponent transport model for PHT3D Example 11."""

from __future__ import annotations

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


NLAY = 58
NROW = 1
NCOL = 99
NXYZ = NLAY * NROW * NCOL

DELR = np.array(
    [1.0] * 7
    + [0.5] * 2
    + [0.25] * 76
    + [0.5] * 2
    + [1.0] * 12,
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
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=86.4,
        k33=86.4,
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
        for layer, rate in zip(range(8, NLAY), left_rates)
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

    recharge = [
        [(0, 0, col), 1.0e-3, *recharge_concentrations] for col in range(NCOL)
    ]
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
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=False,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            complexity="MODERATE",
            outer_dvclose=1.0e-7,
            outer_maximum=100,
            inner_maximum=100,
            inner_dvclose=1.0e-8,
            rcloserecord=1.0e-7,
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
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=0.30,
            filename=f"{gwt_name}.mst",
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
        cnc_data = [
            [(layer, 0, col), float(ambient_concentrations[index])]
            for layer in range(NLAY)
            for col in range(6)
            for index, name in enumerate(species_list)
            if name == species_name
        ]
        flopy.mf6.ModflowGwtcnc(
            gwt,
            pname="CNC-UPSTREAM",
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
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )

    sim.write_simulation(silent=True)
