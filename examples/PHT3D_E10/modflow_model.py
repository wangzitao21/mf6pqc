import flopy
import numpy as np

def transport_model(
    perlen=500,
    nstp=50,
    initial_head=None,
    sim_ws=None,
    species_list=None,
    initial_conc=None,
    inflow_concentrations=None,
    hk=None,
    mf6_exe="./bin/mf6.7.0/mf6.exe",
):
    """Build the MODFLOW 6 flow and component-transport models for Example 10."""
    sim_name = "model"
    gwfname = f"gwf_{sim_name}"

    Lx, Ly = 200.0, 50.0
    nlay, nrow, ncol = 1, 40, 80

    delr = Lx / ncol
    delc = Ly / nrow

    top = 10.0
    botm = 0.0

    nper = 1
    tsmult = 1.0
    
    sim = flopy.mf6.MFSimulation(
        sim_name=sim_name,
        sim_ws=sim_ws,
        exe_name=mf6_exe,
        verbosity_level=0
    )

    # Match the 50 external 10-day flow/chemistry intervals. The shorter MMOC
    # particle steps reported by PHT3D are internal to its advection solve and
    # are not additional PHREEQC coupling steps.
    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)]
    )

    gwf_model = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=False)

    ims = flopy.mf6.ModflowIms(
        sim,
        pname='ims',
        complexity='SIMPLE',
        outer_dvclose=1.0e-8,
        outer_maximum=50,
        under_relaxation='NONE',
        inner_maximum=100,
        inner_dvclose=1.0e-9,
        rcloserecord=1.0e-10,
        linear_acceleration='CG',
        scaling_method='NONE',
        reordering_method='NONE',
        relaxation_factor=0.97
    )
    sim.register_ims_package(ims, [gwf_model.name])

    flopy.mf6.ModflowGwfdis(
        gwf_model,
        pname='dis',
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm
    )

    flopy.mf6.ModflowGwfnpf(
        gwf_model,
        pname='npf',
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=hk,
    )

    if initial_head is None:
        initial_head = np.full((nlay, nrow, ncol), 5.0)

    flopy.mf6.ModflowGwfic(
        gwf_model,
        pname='ic',
        strt=initial_head
    )

    flopy.mf6.ModflowGwfsto(
        gwf_model,
        pname='sto',
        save_flows=False,
        iconvert=1,
        ss=0.00,
        sy=0.00,
        steady_state={0: True},
    )

    chd_spd_1 = []
    for i in range(nrow):
        chd_spd_1.append([(0, i, 0), 5.0, *inflow_concentrations])
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='chd-1',
        save_flows=False,
        maxbound=len(chd_spd_1),
        stress_period_data={0: chd_spd_1},
        auxiliary=species_list,
        filename=f"{gwfname}.1.chd"
    )

    chd_spd_2 = []
    for i in range(nrow):
        chd_spd_2.append([(0, i, ncol-1), 3.0])
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='chd-2',
        save_flows=False,
        maxbound=len(chd_spd_2),
        stress_period_data={0: chd_spd_2},
        filename=f"{gwfname}.2.chd"
    )

    flopy.mf6.ModflowGwfoc(
        gwf_model,
        pname='oc',
        budget_filerecord=f'{gwfname}.bud',
        head_filerecord=f'{gwfname}.hds',
        saverecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )

    # Split the component-major PhreeqcRM buffer into one field per GWT model.
    species_conc = {}
    for i in range(len(species_list)):
        start = i * nlay * nrow * ncol
        end = (i + 1) * nlay * nrow * ncol
        species_conc[species_list[i]] = initial_conc[start:end]

    gwt_models = {}
    for species_name, species_initial_conc in species_conc.items():

        # Official Example 10 dispersion parameters.
        alh = 0.5
        ath1 = 0.1
        # The official Ex10 DSP input specifies zero effective molecular
        # diffusion; at this scale 3e-10 was negligible but not identical.
        diffc = 0.0
        porosity = 0.30

        gwtname = f"gwt_{species_name}_model"
        gwt_model = flopy.mf6.ModflowGwt(
            sim, 
            modelname=gwtname,
            save_flows=False, 
            model_nam_file=f"{gwtname}.nam"
        )

        imsgwt = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=1.0e-8,
            outer_maximum=100,
            under_relaxation="NONE",
            inner_maximum=300,
            inner_dvclose=1.0e-10,
            rcloserecord=1.0e-8,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            filename=f"{gwtname}.ims"
        )
        sim.register_ims_package(imsgwt, [gwt_model.name])
        
        # 创建离散化包
        flopy.mf6.ModflowGwtdis(
            gwt_model, 
            nlay=gwf_model.dis.nlay.get_data(), 
            nrow=gwf_model.dis.nrow.get_data(), 
            ncol=gwf_model.dis.ncol.get_data(), 
            delr=gwf_model.dis.delr.array, 
            delc=gwf_model.dis.delc.array, 
            top=gwf_model.dis.top.array,
            botm=gwf_model.dis.botm.array, 
            idomain=1, 
            filename=f"{gwtname}.dis"
        )

        flopy.mf6.ModflowGwtic(gwt_model, strt=species_initial_conc, filename=f"{gwtname}.ic")
        
        # PHT3D uses MMOC. MODFLOW 6 has no particle-tracking MMOC option;
        # TVD is its closest high-resolution Eulerian counterpart.
        flopy.mf6.ModflowGwtadv(gwt_model, scheme="TVD", filename=f"{gwtname}.adv")
        
        flopy.mf6.ModflowGwtdsp(
            gwt_model, 
            xt3d_off=True, 
            alh=alh, #alv=alv,
            ath1=ath1, #atv=atv,
            diffc=diffc,
            filename=f"{gwtname}.dsp"
        )

        flopy.mf6.ModflowGwtmst(gwt_model, porosity=porosity, filename=f"{gwtname}.mst")

        # The upstream CHD is the inflow boundary. Its auxiliary component
        # values are transferred into GWT by SSM.
        sourcerecarray = [("chd-1", "AUX", species_name)]
        flopy.mf6.ModflowGwtssm(
            gwt_model, 
            pname=f'{species_name}_ssm',
            sources=sourcerecarray, 
            filename=f"{gwtname}.ssm"
        )
        
        flopy.mf6.ModflowGwtoc(
            gwt_model, 
            budget_filerecord=f"{gwtname}.cbc", 
            concentration_filerecord=f"{gwtname}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")]
        )
        
        flopy.mf6.ModflowGwfgwt(
            sim, 
            exgtype="GWF6-GWT6", 
            exgmnamea=gwfname, 
            exgmnameb=gwtname, 
            filename=f"{gwtname}.gwfgwt"
        )
        
        gwt_models[species_name] = gwt_model

    sim.write_simulation(silent=False)
