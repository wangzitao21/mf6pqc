import flopy
import numpy as np
import os
from flopy.utils import CellBudgetFile

from utils import map_values_to_domain

def transport_model(
    sim_ws='./simulation/mycase',
    species_list=["Ca", "Mg", "Cl"],
    perlen=365,
    nstp=365,
    initial_conc=np.ones(120000) * 0.05,
    bc=[0.1, 0.1, 0.1],
    lake=[0.0, 0.0, 0.0],
    porosity=0.30,
    K11=1.0,
    initial_density=1000.0,
    initial_head=100.0
):

    gwf_model_name = 'gwf_model'
    
    nlay, nrow, ncol = 100, 400, 1
    delr = [60.0]
    delc = [1.0] * nrow

    top = 60.0
    botm = np.arange(59.4, -0.6, -0.6)
    botm[-1] = 0

    # top = 60.0
    # thickness = 0.6
    # botm_list = (np.linspace(top - thickness, 0.0, num=nlay)).tolist()

    # botm_list[10] = botm_list[10] - (4.2 - 0.6) # 让第11层厚度变为4.2m
    # del botm_list[11:17] # 移除被合并的6层(原12-17层)
    # botm_final = np.array(botm_list)
    # nlay_final = len(botm_final)  
    # botm =   botm_list
    # nlay = nlay_final


    nper = 1
    tsmult = 1.0
    
    sim = flopy.mf6.MFSimulation(
        sim_name="model", # 总的模拟名
        sim_ws=sim_ws,    # 模拟文件夹
        exe_name='mf6',
        verbosity_level=2
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen, nstp, tsmult)]
    )

    gwf_model = flopy.mf6.ModflowGwf(sim, modelname=gwf_model_name, save_flows=False)

    # GWF IMS (地下水流模型求解器) - 极度宽松设置
    # ims = flopy.mf6.ModflowIms(
    #     sim,
    #     pname='ims',
    #     print_option="SUMMARY",      # 保持输出简洁
    #     complexity='SIMPLE',         # 使用最简单的默认设置起点（虽然会被覆盖）
    #     outer_dvclose=0.5,           # **极大放宽**外循环水头收敛标准 (允许每次迭代 0.5m 的变化)
    #     outer_maximum=100,           # 适当减少最大外循环次数，避免无效振荡太久
    #     under_relaxation="NONE",     # 禁用欠松弛，保持简单
    #     backtracking_number=0,       # 禁用回溯，保持简单
    #     inner_maximum=50,            # 适当减少最大内循环次数
    #     inner_dvclose=0.1,           # **极大放宽**内循环水头收敛标准 (但仍比 outer_dvclose 严格一个数量级)
    #     rcloserecord=100.0,          # **极大放宽**绝对流量残差标准 (单位 L³/T，比如 m³/day)
    #     linear_acceleration='BICGSTAB',
    #     scaling_method="NONE",       # 禁用缩放
    #     reordering_method="NONE"     # 禁用重排序
    # )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname='ims',
        print_option="SUMMARY",
        complexity='COMPLEX',            # 对于非线性问题，使用 'COMPLEX'
        outer_maximum=200,               # 增加最大外迭代次数
        outer_dvclose=1.0e-4,            # 放宽水头收敛标准
        inner_maximum=100,               # 增加最大内迭代次数
        inner_dvclose=1.0e-5,            # 放宽内循环收敛标准
        rcloserecord=1.0e-1,             # 放宽流量残差标准
        under_relaxation='DBD',          # 使用 DBD 欠松弛来稳定收敛
        under_relaxation_theta=0.7,
        under_relaxation_kappa=0.1,
        under_relaxation_gamma=0.0,
        under_relaxation_momentum=0.0,
        backtracking_number=20,          # 允许回溯
        backtracking_tolerance=1.1,
        backtracking_reduction_factor=0.7,
        backtracking_residual_limit=100
    )
    sim.register_ims_package(ims, [gwf_model.name])

    idomain = np.genfromtxt("./input_data/idomain.txt", delimiter='\t')
    idomain = idomain.astype(int) # 确保 idomain 是整数类型，以便进行比较

    flopy.mf6.ModflowGwfdis(
        gwf_model,
        pname='dis',
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=idomain
    )

    k11 = np.genfromtxt("./input_data/K11.txt", delimiter='\t')
    flopy.mf6.ModflowGwfnpf(
        gwf_model,
        pname='npf',
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=k11,
        # k22=hk*0.1,
        k33=k11*0.1,
        rewet_record= ['wetfct', 0.5, 'iwetit', 1, 'ihdwet', 0],
        wetdry=0.1,
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
        ss=1.0e-5,
        sy=0.15
    )

    # ! ------ 相同 ------
    extend_species_list = ["density"]
    extend_species_list.extend(species_list)

    # ! ------ 补水渠 ------
    bs_data = np.loadtxt("./input_data/bs.txt")
    bs_data = bs_data[:, [0, 1, 2]] - 1
    bs_spd = []
    for i in np.arange(bs_data.shape[0]):
        bs_spd.append([tuple(bs_data[i].astype(int)), 60.0, 400.0, 55.0, 1198.844, *bc])
    flopy.mf6.ModflowGwfriv(
        gwf_model,
        pname='bs',
        save_flows=False,
        maxbound=len(bs_spd),
        stress_period_data={0: bs_spd},
        auxiliary=extend_species_list,
        filename=f"{gwf_model_name}.bs.riv"
    )

    # ! ------ 达布逊湖 ------
    lake_data = np.loadtxt("./input_data/lake.txt")
    lake_data = lake_data[:, [0, 1, 2]] - 1
    lake_spd = []
    for i in np.arange(lake_data.shape[0]):
        lake_spd.append([tuple(lake_data[i].astype(int)), 60.0, 400.0, 55.0, 1147.8, *lake])
    flopy.mf6.ModflowGwfriv(
        gwf_model,
        pname='lake',
        save_flows=False,
        maxbound=len(lake_spd),
        stress_period_data={0: lake_spd},
        auxiliary=extend_species_list,
        filename=f"{gwf_model_name}.lake.riv",
    )

    # ! ------ 上游补给 ------
    shangyou_data = np.loadtxt("./input_data/shangyou.txt")
    shangyou_data = shangyou_data[:, [0, 1, 2]] - 1
    shangyou_spd = []
    for i in np.arange(shangyou_data.shape[0]):
        shangyou_spd.append([tuple(shangyou_data[i].astype(int)), 63.0, 400.0, 1050.0, *bc])
    flopy.mf6.ModflowGwfghb(
        gwf_model,
        pname='shangyou',
        save_flows=False,
        maxbound=len(shangyou_spd),
        stress_period_data={0: shangyou_spd},
        auxiliary=extend_species_list,
        filename=f"{gwf_model_name}.shangyou.ghb",
    )

    # ! ------ 抽水井 ------
    cl_data = np.loadtxt("./input_data/cl.txt")
    cl_data = cl_data[:, [0, 1, 2]] - 1
    cl_spd = []
    for i in np.arange(cl_data.shape[0]):
        cl_spd.append([tuple(cl_data[i].astype(int)), -10.0])
    cl_spd = cl_spd
    flopy.mf6.ModflowGwfwel(
        gwf_model,
        pname='cl',
        save_flows=False,
        maxbound=len(cl_spd),
        stress_period_data={0: cl_spd},
        filename=f"{gwf_model_name}.cl.wel"
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

    initial_conc = map_values_to_domain(initial_conc.reshape(10, 23965), idomain, fill_value=0.0)
    
    # ! 将初始 phreeqcrm 数组转换成字典格式
    species_conc = {}
    for i in range(len(species_list)):
        species_conc[species_list[i]] = initial_conc[i]

    gwt_models = {}
    for species_name, species_initial_conc in species_conc.items():

        alh = 10.0    # 纵向弥散度 使用 alh 和 alv 设置纵向弥散系数
        ath1 = alh / 10 # 使用 ath1, ath2, atv 设置横向弥散系数
        alv = alh / 10
        diffc = 0.1 # 分子扩散系数

        gwt_model_name = f"gwt_{species_name}_model"
        gwt_model = flopy.mf6.ModflowGwt(sim, modelname=gwt_model_name, save_flows=False, 
                                  model_nam_file=f"{gwt_model_name}.nam")

        # GWT IMS (各溶质运移模型求解器) - 极度宽松设置
        # imsgwt = flopy.mf6.ModflowIms(
        #     sim,
        #     print_option="SUMMARY",
        #     outer_dvclose=0.1,
        #     outer_maximum=100,
        #     under_relaxation="NONE",
        #     inner_maximum=50,
        #     inner_dvclose=0.01,
        #     rcloserecord=10.0,
        #     linear_acceleration="BICGSTAB",
        #     scaling_method="NONE",
        #     reordering_method="NONE",
        #     filename=f"{gwt_model_name}.ims"
        # )
        # sim.register_ims_package(imsgwt, [gwt_model.name])

        nouter, ninner = 50, 100
        hclose, rclose, relax = 1e-6, 1e-6, 1.0
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
            idomain=idomain, 
            filename=f"{gwt_model_name}.dis"
        )

        flopy.mf6.ModflowGwtic(gwt_model, strt=species_initial_conc, filename=f"{gwt_model_name}.ic")

        flopy.mf6.ModflowGwtadv(gwt_model, scheme="TVD", filename=f"{gwt_model_name}.adv")
        
        flopy.mf6.ModflowGwtdsp(
            gwt_model, 
            xt3d_off=True, 
            alh=alh, alv=alv,
            ath1=ath1, #atv=atv,
            diffc=diffc,
            filename=f"{gwt_model_name}.dsp"
        )

        flopy.mf6.ModflowGwtmst(gwt_model, porosity=porosity, filename=f"{gwt_model_name}.mst")
        
        sourcerecarray = [("lake", "AUX", species_name),
                          ("bs", "AUX", species_name),
                          ("shangyou", "AUX", species_name),
                          ]
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

# ! ######################### 单独密度模型 ######################### ! #

    gwt_density_model_name = "gwt_density"
    
    gwt_density_model = flopy.mf6.ModflowGwt(sim, modelname=gwt_density_model_name, save_flows=False,
                                       model_nam_file=f"{gwt_density_model_name}.nam")
    
    # imsgwt_density = flopy.mf6.ModflowIms(
    #     sim,
    #     print_option="SUMMARY",
    #     outer_dvclose=1.0,           # **极大放宽**外循环密度收敛标准 (允许每次迭代变化 1.0 kg/m³)
    #     outer_maximum=150,
    #     # 保留 DBD 和 Backtracking 可能有助于稳定这个关键的非线性部分
    #     under_relaxation="DBD",
    #     under_relaxation_theta=0.7,
    #     under_relaxation_kappa=0.1,
    #     under_relaxation_gamma=0.1,
    #     under_relaxation_momentum=0.001,
    #     backtracking_number=5,       # 减少回溯次数
    #     backtracking_tolerance=10.0, # 放宽回溯容差
    #     backtracking_reduction_factor=0.3,
    #     backtracking_residual_limit=500, # 提高回溯残差限制
    #     inner_maximum=75,            # 稍微多给点机会
    #     inner_dvclose=0.1,           # **极大放宽**内循环密度收敛标准
    #     rcloserecord=50.0,           # **极大放宽**绝对密度通量残差标准
    #     linear_acceleration="BICGSTAB",
    #     scaling_method="NONE",
    #     reordering_method="NONE",
    #     filename=f"{gwt_density_model_name}.ims"
    # )
    # sim.register_ims_package(imsgwt_density, [gwt_density_model.name])

    imsgwt_density = flopy.mf6.ModflowIms(
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
        filename=f"{gwt_density_model_name}.ims"
    )
    sim.register_ims_package(imsgwt_density, [gwt_density_model.name])

    flopy.mf6.ModflowGwtdis(gwt_density_model, nlay=nlay, nrow=nrow, ncol=ncol,
                             delr=delr, delc=delc, top=top, botm=botm,
                             idomain=idomain, filename=f"{gwt_density_model_name}.dis")

    initial_density = map_values_to_domain(initial_density, idomain, fill_value=0.0)
    flopy.mf6.ModflowGwtic(gwt_density_model, strt=initial_density, filename=f"{gwt_density_model_name}.ic")
    
    flopy.mf6.ModflowGwtadv(gwt_density_model, scheme="TVD", filename=f"{gwt_density_model_name}.adv")
    flopy.mf6.ModflowGwtdsp(gwt_density_model, xt3d_off=True, alh=alh, ath1=ath1, diffc=diffc,
                             filename=f"{gwt_density_model_name}.dsp")
    flopy.mf6.ModflowGwtmst(gwt_density_model, porosity=porosity, filename=f"{gwt_density_model_name}.mst")
    flopy.mf6.ModflowGwtssm(gwt_density_model, pname="density_ssm",
                            sources=[("lake", "AUX", "density"),
                                     ("bs", "AUX", "density"),
                                     ("shangyou", "AUX", "density"),],
                            filename=f"{gwt_density_model_name}.ssm")
    flopy.mf6.ModflowGwtoc(gwt_density_model, budget_filerecord=f"{gwt_density_model_name}.cbc", concentration_filerecord=f"{gwt_density_model_name}.ucn",
                            saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "ALL")])
    flopy.mf6.ModflowGwfgwt(sim, exgtype="GWF6-GWT6", exgmnamea=gwf_model_name,
                            exgmnameb=gwt_density_model_name, filename=f"{gwt_density_model_name}.gwfgwt")

    buy_packagedata = [
        (0, 1.0, 1000.0, gwt_density_model_name, None) # (drhodc=1.0, crhoref=0.0, modelname="density")
    ]
    flopy.mf6.ModflowGwfbuy(gwf_model, 
                            denseref=1000,
                            nrhospecies=len(buy_packagedata), # 应该是 1
                            density_filerecord=['model_density.bin'],
                            packagedata=buy_packagedata,
                            filename=f"{gwf_model_name}.buy"
                           )

# ! ######################### 写入和运行模型 ######################### ! #

    sim.write_simulation(silent=False)
    sim.run_simulation(silent=False, report=True)

# ! ######################### 读取和输出结果 ######################### ! #

    # head = gwf_model.oc.output.head().get_alldata()

    # concentration_data = []
    # for species, gwt_model in gwt_models.items():
    #     concentration_data.append(gwt_model.oc.output.concentration().get_alldata().ravel())

    # ! 展平还给 phreeqcrm
    # concentration_data = np.array(concentration_data).ravel()

    # density_value = gwt_density_model.oc.output.concentration().get_alldata().ravel()
    # density_value = np.array(density_value)

    # 打开预算文件
    # bud = gwf.output.budget()  # 不需要手动给文件名
    # spdis = bud.get_data(text='DATA-SPDIS')[0]  # dict，包含 'q': ndarray

    # # 得到三个方向的比速分量 array，shape=(nlay, nrow, ncol)
    # qx, qy, qz = flopy.utils.postprocessing.get_specific_discharge(spdis, gwf)

    # # 如果想要孔隙水速，就除以 porosity
    # vx = qx / porosity
    # vy = qy / porosity
    # vz = qz / porosity
    
    # return head, concentration_data

# density_data = np.load("./input_data/hk_arcsin1.npy")
# density_data = (density_data - density_data.min()) / (density_data.max() - density_data.min()) * (1200 - 1100) + 1100

# head, concentrations = transport_model(species_list=["Ca", "Mg", "Cl"],
#                                  perlen=100,
#                                  nstp=10,
#                                  initial_conc=np.ones(120000) * 0.05,
#                                  bc=[0.1, 0.1, 0.1],
#                                  porosity=0.35,
#                                  K11=1.0,
#                                 initial_density=1000.0,
#                                 initial_head=100.0
#                                  )
# # print(head.shape)

# np.save("head.npy", head)

