"""MODFLOW 6 flow and transport model for PHT3D Example 3."""

from __future__ import annotations

from pathlib import Path

import flopy


NLAY = 1
NROW = 1
NCOL = 80
NXYZ = NLAY * NROW * NCOL

CELL_LENGTH = 0.005
POROSITY = 0.35
LONGITUDINAL_DISPERSIVITY = 0.005
SIMULATION_TIME = 24.0
TIME_STEPS = 192
INFLOW_RATE = 0.007


def transport_model(
    *,
    sim_ws: str | Path,
    species_list: list[str],
    initial_conc,
    inflow_concentrations,
    mf6_exe: str | Path,
) -> flopy.mf6.MFSimulation:
    """Write the one-dimensional benchmark flow and transport simulation."""
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=sim_ws,
        exe_name=str(mf6_exe),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(SIMULATION_TIME, TIME_STEPS, 1.0)],
    )

    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_dvclose=1.0e-10,
        outer_maximum=50,
        inner_maximum=100,
        inner_dvclose=1.0e-10,
        rcloserecord=1.0e-10,
        linear_acceleration="CG",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(flow_ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=CELL_LENGTH,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=0.056,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )

    # These are the MODFLOW-2000 boundaries used to create the official
    # MT3D.FLO file: a 0.007 m3/d well in cell 1 and fixed head in cell 80.
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-1",
        save_flows=True,
        stress_period_data={
            0: [[(0, 0, 0), INFLOW_RATE, *inflow_concentrations]]
        },
        auxiliary=species_list,
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-OUTFLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, NCOL - 1), 1.0]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    species_concentrations = {
        name: initial_conc[index * NXYZ : (index + 1) * NXYZ]
        for index, name in enumerate(species_list)
    }

    for species_name, species_initial_conc in species_concentrations.items():
        gwt_name = f"gwt_{species_name}_model"
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=True,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=1.0e-10,
            outer_maximum=50,
            inner_maximum=100,
            inner_dvclose=1.0e-10,
            rcloserecord=1.0e-10,
            linear_acceleration="BICGSTAB",
            relaxation_factor=1.0,
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])

        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=CELL_LENGTH,
            delc=1.0,
            top=1.0,
            botm=0.0,
            filename=f"{gwt_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            gwt,
            strt=species_initial_conc,
            filename=f"{gwt_name}.ic",
        )
        # PHT3D uses MMOC, which GWT does not provide.  CENTRAL is stable for
        # this grid (cell Peclet number = delr/alh = 1) and gave the closest
        # match among GWT's available schemes.
        flopy.mf6.ModflowGwtadv(
            gwt,
            scheme="CENTRAL",
            filename=f"{gwt_name}.adv",
        )
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=LONGITUDINAL_DISPERSIVITY,
            ath1=0.1 * LONGITUDINAL_DISPERSIVITY,
            diffc=0.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=POROSITY,
            filename=f"{gwt_name}.mst",
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=[("WEL-1", "AUX", species_name)],
            filename=f"{gwt_name}.ssm",
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
    return sim
