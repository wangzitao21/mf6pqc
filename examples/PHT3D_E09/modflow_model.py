import flopy
import numpy as np


def transport_model(
    *,
    sim_ws,
    species_list,
    initial_conc,
    background_concentrations,
    well_concentrations,
    mf6_exe,
    perlen=500.0,
    nstp=50,
):
    """Build the MODFLOW 6 flow and transport models for PHT3D Example 9."""
    sim_name = "model"
    gwf_name = f"gwf_{sim_name}"

    nlay, nrow, ncol = 1, 31, 51
    lx, ly = 510.0, 310.0

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

    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_dvclose=1.0e-8,
        outer_maximum=50,
        inner_maximum=100,
        inner_dvclose=1.0e-9,
        # The original MODFLOW-2000 PCG model uses hclose=rclose=1e-3.
        # A 1e-6 flow residual tolerance is already substantially tighter and
        # avoids false non-convergence reports at machine-precision residuals.
        rcloserecord=1.0e-6,
        linear_acceleration="CG",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(flow_ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=lx / ncol,
        delc=ly / nrow,
        top=10.0,
        botm=0.0,
    )
    initial_head = np.full((nlay, nrow, ncol), 99.0, dtype=float)
    initial_head[:, :, 0] = 100.0
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=50.0,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        iconvert=1,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )

    left_chd = [
        [(0, row, 0), 100.0, *background_concentrations]
        for row in range(nrow)
    ]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-LEFT",
        save_flows=True,
        stress_period_data={0: left_chd},
        auxiliary=species_list,
        filename=f"{gwf_name}.left.chd",
    )
    right_chd = [[(0, row, ncol - 1), 99.0] for row in range(nrow)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-RIGHT",
        save_flows=True,
        stress_period_data={0: right_chd},
        filename=f"{gwf_name}.right.chd",
    )

    well_data = [[(0, 15, 15), 2.0, *well_concentrations]]
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-1",
        save_flows=True,
        stress_period_data={0: well_data},
        auxiliary=species_list,
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    cell_count = nlay * nrow * ncol
    species_conc = {
        name: initial_conc[index * cell_count : (index + 1) * cell_count]
        for index, name in enumerate(species_list)
    }

    for species_name, species_initial_conc in species_conc.items():
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
            outer_dvclose=1.0e-6,
            outer_maximum=50,
            inner_maximum=100,
            inner_dvclose=1.0e-6,
            rcloserecord=1.0e-6,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])

        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=nlay,
            nrow=nrow,
            ncol=ncol,
            delr=lx / ncol,
            delc=ly / nrow,
            top=10.0,
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
            alh=10.0,
            ath1=3.0,
            atv=1.0,
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
                ("WEL-1", "AUX", species_name),
                ("CHD-LEFT", "AUX", species_name),
            ],
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
