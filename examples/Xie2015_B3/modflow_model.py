import flopy
import numpy as np
import matplotlib.pyplot as plt
import os
from flopy.utils import CellBudgetFile

def transport_model(
    
    nrow=1,
    ncol=80,
    nlay=1,

    sim_ws="",
    species_list=["Ca", "Mg", "Cl"],
    perlen=365*10,
    nstp=1000*10,
    initial_conc=np.ones(120000) * 0.05,
    bc=[0.1, 0.1, 0.1],
    porosity=0.35,
    K11=10.0,
    initial_head=0.0
):

    gwf_model_name = 'gwf_model'

    delr = [0.025] * ncol
    delc = [1.0]
    top = 1.0
    botm = 0

    nper = 1
    tsmult = 1.0

    hk = K11
    
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=sim_ws,
        exe_name='./bin/mf6.exe',
        verbosity_level=0
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)]
    )

    gwf_model = flopy.mf6.ModflowGwf(
        sim, 
        modelname=gwf_model_name, 
        save_flows=False
    )
    
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

    # chd_spd = [[(0, 0, ncol-1),0.00004375]] # 0.0
    # flopy.mf6.ModflowGwfchd(
    #     gwf_model,
    #     pname='chd',
    #     save_flows=True,
    #     maxbound=len(chd_spd),
    #     stress_period_data={0: chd_spd},
    #     filename=f"{gwf_model_name}.choushui.chd"
    # )

    # chd2_spd = [[(0, 0, 0), 0.00695625, *bc],] # 0.007
    # flopy.mf6.ModflowGwfchd(
    #     gwf_model,
    #     pname='bushui',
    #     save_flows=True,
    #     maxbound=len(chd2_spd),
    #     stress_period_data={0: chd2_spd},
    #     auxiliary=species_list,
    #     filename=f"{gwf_model_name}.bushui.chd"
    # )

    ghb_spd = [[(0, 0, ncol-1), 0.0, 800.0]] 
    flopy.mf6.ModflowGwfghb(
        gwf_model,
        pname='ghb_right',
        save_flows=True,
        maxbound=len(ghb_spd),
        stress_period_data={0: ghb_spd},
        filename=f"{gwf_model_name}.choushui.ghb"
    )

    ghb2_spd = [[(0, 0, 0), 0.007, 800.0, *bc]] 
    flopy.mf6.ModflowGwfghb(
        gwf_model,
        pname='bushui',
        save_flows=True,
        maxbound=len(ghb2_spd),
        stress_period_data={0: ghb2_spd},
        auxiliary=species_list,
        filename=f"{gwf_model_name}.bushui.ghb"
    )

    flopy.mf6.ModflowGwfoc(
        gwf_model,
        pname='oc',
        budget_filerecord=f'{gwf_model_name}.bud',
        head_filerecord=f'{gwf_model_name}.hds',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'LAST')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
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
    alh = 0.0    # 纵向弥散度 使用 alh 和 alv 设置纵向弥散系数
    ath1 = alh / 10 # 使用 ath1, ath2, atv 设置横向弥散系数
    diffc = 0.0 # 分子扩散系数

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

        flopy.mf6.ModflowGwtadv(gwt_model, scheme="TVD", filename=f"{gwt_model_name}.adv")
        
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
        
        sourcerecarray = [("bushui", "AUX", species_name)]
        flopy.mf6.ModflowGwtssm(
            gwt_model, 
            pname=f'{species_name}_ssm',
            sources=sourcerecarray, 
            filename=f"{gwt_model_name}.ssm"
        )
        
        flopy.mf6.ModflowGwtoc(
            gwt_model, 
            budget_filerecord=f"{gwt_model_name}.cbc", 
            concentration_filerecord=f"{gwt_model_name}.ucn",
            budgetcsv_filerecord="{}.oc.csv".format(gwt_model_name),
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

# ! ######################### 更新密度 ######################### ! #
    # buy_packagedata = [
    #     (0, 24.8171, 0.0, 'gwt_Cl_model', "concentration"),
    #     (1, 27.3688, 0.0, 'gwt_K_model',  "concentration"),
    #     (2, 16.0929, 0.0, 'gwt_Na_model', "concentration"),
    #     (3, 28.0546, 0.0, 'gwt_Ca_model', "concentration"),
    #     (4, 17.0135, 0.0, 'gwt_Mg_model', "concentration"),
    #     (5, 67.2427, 0.0, 'gwt_S_model',  "concentration"),
    #     (6, 42.0056, 0.0, 'gwt_C_model',  "concentration"),
    # ]
    # flopy.mf6.ModflowGwfbuy(
    #     gwf_model, 
    #     denseref=1000.0,
    #     nrhospecies=7, # len(buy_packagedata)
    #     density_filerecord=['model_density.bin'],
    #     packagedata=buy_packagedata,
    #     filename=f"{gwf_model_name}.buy"
    # )

# ! ######################### 写入和运行模型 ######################### ! #
    sim.write_simulation(silent=False)