import flopy
import numpy as np
import matplotlib.pyplot as plt
import os
from flopy.utils import CellBudgetFile

def transport_model(perlen=500,
                    nstp=50,
                    initial_head=100.0,
                    sim_ws=None,
                    species_list=["Ca", "Mg", "Cl"],
                    initial_conc=np.ones(120000) * 0.05,
                    bc_1=[0.1, 0.1, 0.1],
                    bc_15=[0.3, 0.2, 0.1],
                   ):
    sim_name = 'model'
    gwf_model_name = 'gwf_model'

    Lx, Ly = 200.0, 50.0
    nlay, nrow, ncol = 1, 40, 80

    delr = Lx / ncol
    delc = Ly / nrow

    top = 10.0
    botm = 0.0

    nper = 1
    tsmult = 1.0

    hk = np.load("./input_data/PHT3D_CASE_10/hk.npy").reshape(nlay, nrow, ncol)
    
    sim = flopy.mf6.MFSimulation(
        sim_name=gwf_model_name,    # 总的模拟名
        sim_ws=sim_ws, # 模拟文件夹
        exe_name='mf6',
        verbosity_level=0
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)]
    )

    gwf_model = flopy.mf6.ModflowGwf(sim, modelname=gwf_model_name, save_flows=False)

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
        sy=0.00
    )

    chd_spd_1 = []
    for i in range(nrow):
        chd_spd_1.append([(0, i, 0), 5.0, *bc_1])
    print("species_list", species_list)
    flopy.mf6.ModflowGwfchd(
        gwf_model,
        pname='chd-1',
        save_flows=False,
        maxbound=len(chd_spd_1),
        stress_period_data={0: chd_spd_1},
        auxiliary=species_list,
        filename=f"{gwf_model_name}.1.chd"
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
        filename=f"{gwf_model_name}.2.chd"
    )

    # Well at bottom-left corner
    # wel_spd = [[(0,0,0), 10.0, *bc]]
    # print("井包的输入: ", wel_spd)
    # flopy.mf6.ModflowGwfwel(
    #     gwf_model,
    #     pname='wel',
    #     save_flows=False,
    #     maxbound=len(wel_spd),
    #     stress_period_data={0: wel_spd},
    #     auxiliary=species_list
    # )

    flopy.mf6.ModflowGwfoc(
        gwf_model,
        pname='oc',
        budget_filerecord=f'{gwf_model_name}.bud',
        head_filerecord=f'{gwf_model_name}.hds',
        saverecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )

# ! ######################### 各种离子溶质运移模型 ######################### ! #

    # ! 将输入的 phreeqcrm 的一维数组转换成字典格式
    species_conc = {}
    for i in range(len(species_list)):
        start = i * nlay * nrow * ncol
        end = (i + 1) * nlay * nrow * ncol
        species_conc[species_list[i]] = initial_conc[start:end]

    gwt_models = {}
    m = 0
    for species_name, species_initial_conc in species_conc.items():

        alh = 0.5       # 纵向弥散度 使用 alh 和 alv 设置纵向弥散系数
        ath1 = alh * 0.1 # 使用 ath1, ath2, atv 设置横向弥散系数
        diffc = 3e-10    # 分子扩散系数
        porosity = 0.30

        gwt_model_name = f"gwt_{species_name}_{sim.name.split('_')[1]}"
        gwt_model = flopy.mf6.ModflowGwt(sim, modelname=gwt_model_name, save_flows=False, 
                                  model_nam_file=f"{gwt_model_name}.nam")

        imsgwt = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=0.1,
            outer_maximum=100,
            under_relaxation="NONE",
            inner_maximum=50,
            inner_dvclose=0.01,
            rcloserecord=10.0,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            filename=f"{gwt_model_name}.ims"
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
            filename=f"{gwt_model_name}.dis"
        )
        if species_name == "Benznapl":
            print("TRueTRUETRUE")
            species_initial_conc = np.load("./input_data/PHT3D_CASE_10/init_Benznapl.npy").reshape(nlay, nrow, ncol)
        elif species_name == "Tolunapl":
            print("TRueTRUETRUE")
            species_initial_conc = np.load("./input_data/PHT3D_CASE_10/init_Tolunapl.npy").reshape(nlay, nrow, ncol)
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

        flopy.mf6.ModflowGwtmst(gwt_model, porosity=porosity, filename=f"{gwt_model_name}.mst")

        cnc_spd = []
        for i in range(nrow):
            cnc_spd.append([(0, i, 0), bc_15[m]],)
        m+=1
        # print(cnc_spd)
        flopy.mf6.ModflowGwtcnc(gwt_model, maxbound=len(cnc_spd), stress_period_data={0: cnc_spd}, )#boundnames=True,)
        
        sourcerecarray = [("chd-1", "AUX", species_name)]
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
    sim.run_simulation(silent=False, report=True)

# ! ######################### 读取和输出结果 ######################### ! #

    head = gwf_model.oc.output.head().get_alldata()

    concentration_data = []
    for species, gwt_model in gwt_models.items():
        concentration_data.append(gwt_model.oc.output.concentration().get_alldata().ravel())
    concentration_data = np.array(concentration_data).ravel() # ! 展平还给 phreeqcrm

    return head, concentration_data

# strt = np.load("./input_data/PHT3D_CASE_10/strt.npy").reshape(1, 40, 80)

# head, concentration_data = transport_model(perlen=500,
#                     nstp=1,
#                     initial_head=100.0,
#                     species_list=["Ca", "Mg", "Cl"],
#                     initial_conc=np.ones(120000) * 0.05,
#                     bc_1=[0.1, 0.1, 0.1],
#                     bc_15=[0.3, 0.2, 0.1],
#                    )
# concentration_data = concentration_data.reshape(3, 40, 80)
# print(concentration_data.shape)
# plt.imshow(concentration_data[1], cmap="rainbow")
# plt.colorbar()
# from sklearn.metrics import r2_score
# head_ref = np.load("./input_data/PHT3D_CASE_10/haed_ref.npy")[-1].ravel()
# # plt.scatter(head_ref, head.ravel())
# print("R2: ", r2_score(head_ref, head.ravel()))
# plt.show()