import numpy as np
import matplotlib.pyplot as plt
import flopy

def create_and_run_models(
    sim_ws='./simulation/PHT3D_CASE_8',
    perlen=1120, 
    nstp=56, 
    bc=None,
    bc_0=None,
    species_list=None, 
    initial_conc=None
):

    sim_name = 'model'

    Lx = 510.0
    Ly = 310.0

    # 网格参数
    nrow = 31
    ncol = 51
    nlay = 1

    delr = Lx / ncol  # 列方向网格大小
    delc = Ly / nrow  # 行方向网格大小

    top = 10.0
    botm = 0.0
    
    hk = 50.0
    
    # 时间离散化参数
    nper = 1
    perlen = np.array([perlen])
    nstp = [nstp]

    # 模型名称 - 确保不超过16个字符
    # 截取sim_name以确保总长度不超过16个字符
    # if len(f"gwf_{sim_name}") > 16:
    #     # 如果sim_name包含step，提取步骤号并使用简短格式
    #     if "step" in sim_name:
    #         step_num = sim_name.split("step")[1]
    #         gwfname = f"gwf_s{step_num}"
    #         # 再次检查长度
    #         if len(gwfname) > 16:
    #             # 如果仍然太长，使用更简短的格式
    #             gwfname = f"g{step_num}"
    #     else:
    #         # 如果没有step，简单截断
    #         gwfname = f"gwf_{sim_name[:10]}"
    # else:
    #     gwfname = f"gwf_{sim_name}"

    gwfname = f"gwf_{sim_name}"
    
    # 创建MODFLOW 6模拟
    sim = flopy.mf6.MFSimulation(sim_name=gwfname, sim_ws=sim_ws, exe_name='mf6')
    
    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen[0], nstp[0], 1.0)]  # 第三个参数是tsmult
    )
    
    # 创建地下水流模型
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)
    
    # 创建迭代控制包
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
    sim.register_ims_package(ims, [gwf.name])

    # 创建离散化包
    flopy.mf6.ModflowGwfdis(
        gwf,
        pname='dis',
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
        idomain=1,
    )
    
    strt = np.ones((nlay, nrow, ncol), dtype=np.float32) * 100.0
    strt[0, :, -1] = 99.0
    flopy.mf6.ModflowGwfic(gwf, pname='ic', strt=strt)
    
    # 创建节点属性包
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname='npf',
        save_flows=True,
        icelltype=0,
        k=hk
    )
    
    # 创建存储包
    flopy.mf6.ModflowGwfsto(gwf, pname='sto', save_flows=True, iconvert=1, ss=0.0, sy=0.0)
    
    # 创建常水头边界包
    chd_spd_1 = []
    for i in range(nrow):
        chd_spd_1.append([0, i, 0,      100.0, *bc_0])
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname='chd-1',
        save_flows=True,
        maxbound=len(chd_spd_1),
        stress_period_data={0: chd_spd_1},
        auxiliary=species_list,
        filename=f"{gwfname}.1.chd"
    )

    chd_spd_2 = []
    for i in range(nrow):
        chd_spd_2.append([0, i, ncol-1, 99.0])
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname='chd-2',
        save_flows=True,
        maxbound=len(chd_spd_2),
        stress_period_data={0: chd_spd_2},
        filename=f"{gwfname}.2.chd"
    )

    wel_spd = [[(0, 15, 15), 2.0, *bc]]
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname='WEL-1',
        save_flows=True,
        maxbound=len(wel_spd),
        stress_period_data={0: wel_spd},
        auxiliary=species_list
    )

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname='oc',
        budget_filerecord=f'{gwfname}.bud',
        head_filerecord=f'{gwfname}.hds',
        saverecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )

    species_conc = {}
    for i in range(len(species_list)):
        start = i * nlay * nrow * ncol
        end = (i + 1) * nlay * nrow * ncol
        species_conc[species_list[i]] = initial_conc[start:end]

    # 创建各溶质传输模型
    m=0
    gwt_models = {}
    for species_name, species_initial_conc in species_conc.items():
        # 溶质传输模型通用参数
        nouter, ninner = 50, 100
        hclose, rclose, relax = 1e-6, 1e-6, 0.97
        porosity = 0.3
        # 使用 alh 和 alv 设置纵向弥散系数
        alh = 10.0    # 纵向弥散度
        # 使用 ath1, ath2, atv 设置横向弥散系数
        ath1 =  alh * 0.3
        atv = alh * 0.1
        # 设置分子扩散系数
        diffc = 0.0
        
        # 创建溶质传输模型名称
        gwtname = f"gwt_{species_name}_{sim.name.split('_')[1]}"
        
        # 创建溶质传输模型
        gwt = flopy.mf6.ModflowGwt(
            sim, 
            modelname=gwtname, 
            save_flows=True,         
            model_nam_file=f"{gwtname}.nam"
        )
        
        # 创建迭代控制包
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
            filename=f"{gwtname}.ims"
        )
        sim.register_ims_package(imsgwt, [gwt.name])

        flopy.mf6.ModflowGwtdis(
            gwt, 
            nlay=gwf.dis.nlay.get_data(), 
            nrow=gwf.dis.nrow.get_data(), 
            ncol=gwf.dis.ncol.get_data(), 
            delr=gwf.dis.delr.array, 
            delc=gwf.dis.delc.array, 
            top=gwf.dis.top.array,
            botm=gwf.dis.botm.array, 
            idomain=1, 
            filename=f"{gwtname}.dis"
        )
        
        flopy.mf6.ModflowGwtic(
            gwt,
            strt=species_initial_conc,
            filename=f"{gwtname}.ic"
        )
        
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwtname}.adv")

        flopy.mf6.ModflowGwtdsp(
            gwt, 
            xt3d_off=True, 
            alh=alh,
            ath1=ath1,
            atv=atv,
            diffc=diffc,
            filename=f"{gwtname}.dsp"
        )

        # 创建质量存储包
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwtname}.mst")

        # cnc_spd = []
        # for i in range(nrow):
        #     cnc_spd.append([(0, i, 0), bc_0[m]])
        # m+=1
        # flopy.mf6.ModflowGwtcnc(
        #     gwt,
        #     pname='cnc',
        #     maxbound=len(cnc_spd),
        #     stress_period_data={0: cnc_spd},
        #     filename=f"{gwtname}.cnc"
        # )

        # 创建源汇包
        sourcerecarray = [("WEL-1", "AUX", species_name),
                          ("chd-1", "AUX", species_name)
                        ]
        flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 
            filename=f"{gwtname}.ssm"
        )
        
        # 创建输出控制包
        flopy.mf6.ModflowGwtoc(
            gwt, 
            budget_filerecord=f"{gwtname}.cbc", 
            concentration_filerecord=f"{gwtname}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")]
        )
        
        # 创建GWF-GWT交换包
        flopy.mf6.ModflowGwfgwt(
            sim, 
            exgtype="GWF6-GWT6", 
            exgmnamea=gwfname, 
            exgmnameb=gwtname, 
            filename=f"{gwtname}.gwfgwt"
        )
        
        gwt_models[species_name] = gwt
    
    # 写入输入文件
    sim.write_simulation()
    
    # 运行模型
    success = sim.run_simulation()
    if not success:
        raise Exception('MODFLOW 6 did not terminate normally.')
    
    # 获取结果数据
    head = gwf.oc.output.head().get_alldata()
    # print(head.shape)
    # plt.plot(head[0, 0, 0, :])
    # plt.show()
    concentration_data = []
    species_names = []
    
    for species, gwt in gwt_models.items():
        concentration_data.append(gwt.oc.output.concentration().get_alldata())
        species_names.append(species)

    return head

    # # 如果输入是一维数组，则将结果转换回一维数组格式返回
    # if initial_conc is not None and not isinstance(initial_conc, dict):
    #     # 获取最终的浓度数据
    #     result_array = np.zeros(len(species_list) * ncol)
        
    #     # 将每个溶质的浓度数据填充到结果数组中
    #     for i, species in enumerate(species_list):
    #         # 获取该溶质的浓度数据
    #         species_idx = species_names.index(species)
    #         species_conc_data = concentration_data[species_idx][0][0][0]
            
    #         # 填充到结果数组中
    #         result_array[i * ncol:(i + 1) * ncol] = species_conc_data
        
    #     return result_array
    # else:
    #     # 如果输入是字典或None，返回原始格式的结果
    #     # return concentration_data
    #     return head