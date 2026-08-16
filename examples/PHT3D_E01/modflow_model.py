"""MODFLOW 6 flow and transport model for PHT3D Example 1."""

from __future__ import annotations

import flopy


NLAY = 1
NROW = 1
NCOL = 150
NXYZ = NLAY * NROW * NCOL

POROSITY = 0.25
SIMULATION_TIME = 1826.0
TIME_STEPS = 200


def transport_model(
    *,
    sim_ws,
    species_list,
    initial_conc,
    inflow_concentrations,
    mf6_exe,
):
    """Write the one-dimensional, purely advective benchmark model."""
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=sim_ws,
        exe_name=mf6_exe,
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
        delr=1.0,
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
        k=1.0,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )

    # A 3.725 m head drop over the 149 m between boundary-cell centres gives
    # Darcy flux 0.025 m/d and pore-water velocity 0.1 m/d at porosity 0.25.
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-INFLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, 0), 4.725]]},
        filename=f"{gwf_name}.inflow.chd",
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-OUTFLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, NCOL - 1), 1.0]]},
        filename=f"{gwf_name}.outflow.chd",
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
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])

        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=1.0,
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
        flopy.mf6.ModflowGwtadv(
            gwt,
            scheme="TVD",
            filename=f"{gwt_name}.adv",
        )
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=0.0,
            ath1=0.0,
            atv=0.0,
            diffc=0.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=POROSITY,
            filename=f"{gwt_name}.mst",
        )
        flopy.mf6.ModflowGwtcnc(
            gwt,
            pname="CNC-INFLOW",
            stress_period_data={
                0: [[(0, 0, 0), inflow_concentrations[
                    species_list.index(species_name)
                ]]]
            },
            filename=f"{gwt_name}.cnc",
        )
        # MODFLOW 6 requires an SSM package whenever the flow model contains
        # boundary packages, even when concentration is supplied only by CNC.
        flopy.mf6.ModflowGwtssm(
            gwt,
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
