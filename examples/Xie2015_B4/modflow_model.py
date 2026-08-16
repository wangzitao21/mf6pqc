import flopy
import numpy as np

def transport_model(
    
    nrow=1,
    ncol=81,
    nlay=1,

    sim_ws="",
    species_list=["Ca", "Mg", "Cl"],
    perlen=365.0 * 3000,
    nstp=100 * 3000,
    initial_conc=np.ones(120000) * 0.05,
    bc=[0.1, 0.1, 0.1],
    porosity=0.35,
    d0=1.0e-9 * 86400.0,
    K11=10.0,
    initial_head=0.0
):

    gwf_model_name = 'gwf_model'

    if ncol != 81:
        raise ValueError("B4 requires 81 cells: two boundary half cells and 79 interior cells")
    delr = [0.0125] + [0.025] * 79 + [0.0125]
    delc = [1.0]
    top = 1.0
    botm = 0

    perioddata = [(perlen, nstp, 1.0)]
    nper = len(perioddata)

    hk = K11
    
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=sim_ws,
        exe_name='./bin/mf6.7.0/mf6.exe',
        verbosity_level=0
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=perioddata
    )

    gwf_model = flopy.mf6.ModflowGwf(
        sim, 
        modelname=gwf_model_name, 
        save_flows=True
    )
    
    ims = flopy.mf6.ModflowIms(
        sim,
        pname='ims',
        complexity='SIMPLE',
        outer_dvclose=1.0e-8,
        outer_maximum=50,
        under_relaxation='NONE',
        inner_maximum=500,
        inner_dvclose=1.0e-9,
        rcloserecord=1.0e-8,
        linear_acceleration='BICGSTAB',
        scaling_method='DIAGONAL',
        reordering_method='RCM',
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
        save_flows=False,
        save_specific_discharge=True,
        icelltype=0,
        k=hk,
        # k22=hk*0.1,
        k33=hk*0.1
    )

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
        ss=0.0,
        sy=0.0
    )

    # B4 is diffusion-only. Equal fixed heads at both ends enforce zero
    # hydraulic gradient and therefore suppress advective transport.
    chd_spd = [
        [(0, 0, 0), 0.0],
        [(0, 0, ncol - 1), 0.0],
    ]
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='fixed_heads',
        save_flows=True,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd},
        filename=f"{gwf_model_name}.fixed_heads.chd"
    )

    flopy.mf6.ModflowGwfoc(
        gwf_model,
        pname='oc',
        budget_filerecord=f'{gwf_model_name}.bud',
        head_filerecord=f'{gwf_model_name}.hds',
        saverecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )

# ! ######################### 各种离子溶质运移模型 ######################### ! #

    # ! 将输入的 phreeqcrm 的一维数组转换成字典格式
    species_conc = {}
    for i in range(len(species_list)):
        start = i * nlay * nrow * ncol
        end = (i + 1) * nlay * nrow * ncol
        species_conc[species_list[i]] = initial_conc[start:end]

    nouter, ninner = 50, 100
    hclose, rclose, relax = 1e-6, 1e-6, 1.0
    alh = 0.0
    ath1 = 0.0
    # MODFLOW 6 DIFFC is the pore diffusion coefficient Dp = tau * D0.
    # Xie et al. use tau = porosity**(1/3) and D0 = 1e-9 m2/s.
    diffc = porosity ** (1.0 / 3.0) * d0

    bc_conc_map = dict(zip(species_list, bc))

    gwt_models = {}
    for species_name, species_initial_conc in species_conc.items():

        gwt_model_name = f"gwt_{species_name}_model"
        gwt_model = flopy.mf6.ModflowGwt(sim, modelname=gwt_model_name, save_flows=False, 
                                  model_nam_file=f"{gwt_model_name}.nam")

        imsgwt = flopy.mf6.ModflowIms(
            sim, 
            print_option="SUMMARY", 
            outer_dvclose=hclose, 
            outer_maximum=nouter,
            under_relaxation="NONE", 
            inner_maximum=ninner, 
            inner_dvclose=hclose,
            rcloserecord=rclose, 
            linear_acceleration="BICGSTAB",
            scaling_method="NONE", 
            reordering_method="NONE",
            relaxation_factor=relax, 
            filename=f"{gwt_model_name}.ims"
        )
        sim.register_ims_package(imsgwt, [gwt_model.name])
        
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
            filename=f"{gwt_model_name}.dis"
        )

        flopy.mf6.ModflowGwtic(gwt_model, strt=species_initial_conc, filename=f"{gwt_model_name}.ic")

        # Advection is absent in B4; this package has no numerical effect
        # because the fixed-head gradient is zero.
        flopy.mf6.ModflowGwtadv(gwt_model, scheme="UPSTREAM", filename=f"{gwt_model_name}.adv")
        
        flopy.mf6.ModflowGwtdsp(
            gwt_model, 
            xt3d_off=True, 
            alh=alh, #alv=alv,
            ath1=ath1, #atv=atv,
            diffc=diffc,
            filename=f"{gwt_model_name}.dsp"
        )

        flopy.mf6.ModflowGwtmst(
            gwt_model, 
            pname='mst',
            porosity=porosity, 
            filename=f"{gwt_model_name}.mst")

        # MODFLOW 6 requires an SSM package whenever the coupled flow model
        # has a boundary package. It is intentionally empty here: the CHD
        # faces carry no flow and B4's solute boundary is supplied by CNC.
        flopy.mf6.ModflowGwtssm(
            gwt_model,
            pname='ssm',
            filename=f"{gwt_model_name}.ssm"
        )
        
        # B4 uses first-type solute boundaries at both ends. The left half
        # cell is fixed to SOLUTION 1 and the right half cell to the resident
        # SOLUTION 0 composition.
        current_bc_conc = bc_conc_map[species_name]
        current_right_bc_conc = species_initial_conc[-1]
        cnc_spd_list = [
            ((0, 0, 0), current_bc_conc),
            ((0, 0, ncol - 1), current_right_bc_conc),
        ]
        cnc_spd_dict = {0: cnc_spd_list}

        flopy.mf6.ModflowGwtcnc(
            gwt_model,
            pname='fixed_cnc',
            maxbound=len(cnc_spd_list),
            stress_period_data=cnc_spd_dict,
            save_flows=False,
            print_input=False,
            filename=f"{gwt_model_name}.cnc"
        )
        
        flopy.mf6.ModflowGwtoc(
            gwt_model, 
            budget_filerecord=f"{gwt_model_name}.cbc", 
            concentration_filerecord=f"{gwt_model_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")]
        )
        
        flopy.mf6.ModflowGwfgwt(
            sim, 
            exgtype="GWF6-GWT6", 
            exgmnamea=gwf_model_name, 
            exgmnameb=gwt_model_name, 
            filename=f"{gwt_model_name}.gwfgwt"
        )
        
        gwt_models[species_name] = gwt_model

# ! ######################### 写入和运行模型 ######################### ! #
    sim.write_simulation(silent=False)
