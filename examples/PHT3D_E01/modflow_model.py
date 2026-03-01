import flopy
import numpy as np

def transport_model(
    
    nrow=1,
    ncol=150,
    nlay=1,

    sim_ws="",
    species_list=["Spe",],
    perlen=1826,
    nstp=50,
    initial_conc=np.ones(120000) * 0.05,
    bc=[0.1, 0.1, 0.1],
    porosity=0.25,
    K11=1.0,
    initial_head=1.0
):

    gwf_model_name = 'gwf_model'

    delr = [1.0] * ncol
    delc = [1.0]
    top  = 1.0
    botm = 0

    nper = 1
    tsmult = 1.0

    hk = K11
    
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=sim_ws,
        exe_name="./bin/mf6.6.3/mf6.exe",
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
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hk,
        # k22=hk*0.1,
        # k33=hk*0.1
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
        iconvert=0,
        ss=0.0,
        sy=0.0
    )

    # ghb_spd = [[(0, 0, 0), 4.725, 800.0, *bc]] 
    # flopy.mf6.ModflowGwfghb(
    #     gwf_model,
    #     pname='bushui',
    #     save_flows=True,
    #     maxbound=len(ghb_spd),
    #     stress_period_data={0: ghb_spd},
    #     auxiliary=species_list,
    #     filename=f"{gwf_model_name}.bushui.ghb"
    # )

    chd_spd = [[(0, 0, 0), 4.725, *bc]]
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='bushui',
        save_flows=True,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd},
        auxiliary=species_list,
        filename=f"{gwf_model_name}.bushui.chd"
    )

    # chd2_spd = [[(0, 0, 0), 0.00695625, *bc],] # 0.007
    chd2_spd = [[(0, 0, ncol-1), 1.0],]
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='chushui',
        save_flows=True,
        maxbound=len(chd2_spd),
        stress_period_data={0: chd2_spd},
        filename=f"{gwf_model_name}.chushui.chd"
    )

    flopy.mf6.ModflowGwfoc(
        gwf_model,
        pname='oc',
        budget_filerecord=f'{gwf_model_name}.bud',
        head_filerecord=f'{gwf_model_name}.hds',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')],
        printrecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')]
    )

# ! ######################### 各种离子溶质运移模型 ######################### ! #

    # ! src --------------------------------------------------
    src_data_list = []
    # 遍历所有网格 (Layer, Row, Col) 注意 flopy 使用 0-based 索引
    for k in range(nlay):
        for i in range(nrow):
            for j in range(ncol):
                cellid = (k, i, j)
                # 格式: (cellid, smassrate, [aux], [boundname])
                # 这里只填最基本的: ((k, i, j), 0.0)
                src_data_list.append((cellid, 0.0))
    
    # 确定最大边界数，这对于内存分配非常重要
    src_maxbound = len(src_data_list)
    # ! src --------------------------------------------------

    species_conc = {}
    for i in range(len(species_list)):
        start = i * nlay * nrow * ncol
        end = (i + 1) * nlay * nrow * ncol
        species_conc[species_list[i]] = initial_conc[start:end]

    nouter, ninner = 50, 100
    hclose, rclose, relax = 1e-8, 1e-9, 1.0
    alh = 0.0
    ath1 = 0.0
    atv = 0.0
    diffc = 0.0

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
            ath1=ath1, atv=atv,
            diffc=diffc,
            filename=f"{gwt_model_name}.dsp"
        )

        # ! ---------------------------------------------------------------------
        # 实例化 SRC 包
        # ---------------------------------------------------------------------
        flopy.mf6.ModflowGwtsrc(
            gwt_model,
            pname='SRC',          # 给包起个名字，方便查找
            save_flows=True,      # 建议开启，方便检查注入量
            maxbound=src_maxbound,# 关键：预分配全场内存
            stress_period_data={0: src_data_list}, # 填入所有网格的 0.0 初始值
            filename=f"{gwt_model_name}.src"
        )
        # ! ---------------------------------------------------------------------

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
            # budgetcsv_filerecord="{}.oc.csv".format(gwt_model_name),
            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")]
        )
        
        flopy.mf6.ModflowGwfgwt(
            sim, 
            exgtype="GWF6-GWT6", 
            exgmnamea=gwf_model_name, 
            exgmnameb=gwt_model_name, 
            filename=f"{gwt_model_name}.gwfgwt"
        )
        
        gwt_models[species_name] = gwt_model

    sim.write_simulation(silent=False)