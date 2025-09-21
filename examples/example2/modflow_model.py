import numpy as np
import flopy

def transport_model(
    sim_ws='./simulation/PHT3D_CASE_3',
    perlen=24, 
    nstp=192, 
    bc=None,
    species_list=None, 
    initial_conc=None
):
    """
    创建并运行MODFLOW模型
    
    参数:
        perlen: 时间步长
        nstp: 时间步数
        initial_conc: 初始浓度，可以是字典或一维数组(350,)
    
    返回:
        如果initial_conc是字典，返回head, concentration_data, species_names
        如果initial_conc是一维数组，返回处理后的一维数组
    """
    sim_name = 'model'
    
    nrow = 1
    ncol = 80
    nlay = 1
    delr = [0.005] * ncol
    delc = [1.0]
    top = 1.0
    botm = 0.0
    hk = 0.056
    
    if initial_conc is not None and not isinstance(initial_conc, dict):
        conc_dict = {}
        for i, species in enumerate(species_list):
            # 提取每个溶质的浓度值
            species_values = initial_conc[i * ncol:(i + 1) * ncol]
            conc_dict[species] = species_values
        
        initial_conc = conc_dict

    steady = True

    gwfname = f"gwf_{sim_name}"
    
    # 时间离散化参数
    nper = 1
    perlen = np.array([perlen])
    nstp = [nstp]
    steady = [steady]
    
    # 创建MODFLOW 6模拟
    sim = flopy.mf6.MFSimulation(sim_name=gwfname, sim_ws=sim_ws, exe_name='mf6')
    
    tdis = flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen[0], nstp[0], 1.0)]  # 第三个参数是tsmult
    )
    
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)
    
    ic = flopy.mf6.ModflowIms(
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
    sim.register_ims_package(ic, [gwf.name])
    
    dis = flopy.mf6.ModflowGwfdis(
        gwf,
        pname='dis',
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm
    )
    
    # 创建初始条件包
    ic = flopy.mf6.ModflowGwfic(gwf, pname='ic', strt=1.0)
    
    # 创建节点属性包
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname='npf',
        save_flows=True,
        icelltype=0,
        k=hk
    )
    
    # 创建存储包
    sto = flopy.mf6.ModflowGwfsto(gwf, pname='sto', save_flows=True, iconvert=1, ss=0.0, sy=0.0) # ss=1.0e-5, sy=0.1
    
    # 创建常水头边界包
    chd_spd = [[(0, 0, ncol-1), 1.0]]
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname='chd',
        save_flows=True,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd}
    )
    
    # 创建井包 - 包含四种溶质的辅助变量（pH不作为可传输溶质）
    wel_spd = [[(0, 0, 0), 0.007, *bc]]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        pname='WEL-1',
        save_flows=True,
        maxbound=len(wel_spd),
        stress_period_data={0: wel_spd},
        auxiliary=species_list
    )
    # 'H', 'O', 'Charge', 'C', 'Ca', 'Cl', 'Mg'
    # 创建输出控制包
    oc = flopy.mf6.ModflowGwfoc(
        gwf,
        pname='oc',
        budget_filerecord=f'{gwfname}.bud',
        head_filerecord=f'{gwfname}.hds',
        saverecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')],
        printrecord=[('HEAD', 'LAST'), ('BUDGET', 'LAST')]
    )
    
    # 创建各溶质传输模型
    gwt_models = {}
    for species_name, species_initial_conc in initial_conc.items():
        # 溶质传输模型通用参数
        nouter, ninner = 50, 100
        hclose, rclose, relax = 1e-6, 1e-6, 1.0
        porosity = 0.35
        # 使用 alh 和 alv 设置纵向弥散系数
        alh = 0.005    # 纵向弥散度
        # 使用 ath1, ath2, atv 设置横向弥散系数
        ath1 =  0.0005
        # 设置分子扩散系数
        diffc = 0.0
        
        gwtname = f"gwt_{species_name}_{sim.name.split('_')[1]}"
        
        gwt = flopy.mf6.ModflowGwt(sim, modelname=gwtname, save_flows=True, 
                                  model_nam_file=f"{gwtname}.nam")
        
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
        
        if isinstance(species_initial_conc, (list, np.ndarray)):
            flopy.mf6.ModflowGwtic(gwt, strt=species_initial_conc, filename=f"{gwtname}.ic")
        else:
            flopy.mf6.ModflowGwtic(gwt, strt=species_initial_conc, filename=f"{gwtname}.ic")
        
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwtname}.adv")
        
        flopy.mf6.ModflowGwtdsp(
            gwt, 
            xt3d_off=True, 
            alh=alh, #alv=alv,
            ath1=ath1, #atv=atv,
            diffc=diffc,
            filename=f"{gwtname}.dsp"
        )

        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwtname}.mst")
        
        sourcerecarray = [("WEL-1", "AUX", species_name)]
        flopy.mf6.ModflowGwtssm(
            gwt, 
            sources=sourcerecarray, 
            filename=f"{gwtname}.ssm"
        )
        
        flopy.mf6.ModflowGwtoc(
            gwt, 
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
        
        gwt_models[species_name] = gwt
    
    sim.write_simulation()
    
    # 运行模型
    # success = sim.run_simulation()
    # if not success:
    #     raise Exception('MODFLOW 6 did not terminate normally.')
    
    # # 获取结果数据
    # head = gwf.oc.output.head().get_alldata()
    # concentration_data = []
    # species_names = []
    
    # for species, gwt in gwt_models.items():
    #     concentration_data.append(gwt.oc.output.concentration().get_alldata())
    #     species_names.append(species)
    
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
    #     return np.array(concentration_data).reshape(-1)