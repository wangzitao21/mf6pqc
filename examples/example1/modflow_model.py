import numpy as np
import flopy

def transport_model(
    sim_ws='simulation',
    perlen=0.24, 
    nstp=210, 
    bc=None,
    species_list=None, 
    initial_conc=None
):

    sim_name = "model"
    nrow = 1
    ncol = 50
    nlay = 1
    delr = [0.01] * ncol  # 列方向网格大小
    delc = [1.0]          # 行方向网格大小
    top = 1.0             # 顶部高程
    botm = 0.0            # 底部高程
    hk = 1.0              # 水平渗透系数
    
    
    # 如果initial_conc是一维数组，将其转换为字典格式
    if initial_conc is not None and not isinstance(initial_conc, dict):
        # 将一维数组转换为字典
        conc_dict = {}
        for i, species in enumerate(species_list):
            # 提取每个溶质的浓度值
            species_values = initial_conc[i * ncol:(i + 1) * ncol]
            conc_dict[species] = species_values
        initial_conc = conc_dict

    gwfname = f"gwf_{sim_name}"
    
    nper = 1
    perlen = np.array([perlen])
    nstp = [nstp]
    
    sim = flopy.mf6.MFSimulation(sim_name=gwfname, sim_ws=sim_ws, exe_name='./bin/mf6.exe')
    
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
    
    ic = flopy.mf6.ModflowGwfic(gwf, pname='ic', strt=1.0)
    
    npf = flopy.mf6.ModflowGwfnpf(
        gwf,
        pname='npf',
        save_flows=True,
        icelltype=0,
        k=hk
    )
    
    sto = flopy.mf6.ModflowGwfsto(gwf, pname='sto', save_flows=True, iconvert=1, ss=0.0, sy=0.0) # ss=1.0e-5, sy=0.1
    
    chd_spd = [[(0, 0, ncol-1), 1.0]]
    chd = flopy.mf6.ModflowGwfchd(
        gwf,
        pname='chd',
        save_flows=True,
        maxbound=len(chd_spd),
        stress_period_data={0: chd_spd}
    )
    
    wel_spd = [[(0, 0, 0), 0.259061, *bc]]
    wel = flopy.mf6.ModflowGwfwel(
        gwf,
        pname='WEL-1',
        save_flows=True,
        maxbound=len(wel_spd),
        stress_period_data={0: wel_spd},
        auxiliary=['H', 'O', 'Charge', 'C', 'Ca', 'Cl', 'Mg']
    )

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
        porosity = 0.32
        # 使用 alh 和 alv 设置纵向弥散系数
        alh = 0.0067    # 纵向弥散度
        # 使用 ath1, ath2, atv 设置横向弥散系数
        ath1 =  0.00067
        diffc = 0.0 # 分子扩散系数
        
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
    # success = sim.run_simulation()

    # if not success:
    #     raise Exception('MODFLOW 6 did not terminate normally.')
    
    # head = gwf.oc.output.head().get_alldata()
    # concentration_data = []
    # species_names = []
    
    # for species, gwt in gwt_models.items():
    #     concentration_data.append(gwt.oc.output.concentration().get_alldata())
    #     species_names.append(species)
    
    # if initial_conc is not None and not isinstance(initial_conc, dict):
    #     result_array = np.zeros(len(species_list) * ncol)
        
    #     for i, species in enumerate(species_list):
    #         species_idx = species_names.index(species)
    #         species_conc_data = concentration_data[species_idx][0][0][0]
            
    #         result_array[i * ncol:(i + 1) * ncol] = species_conc_data
        
    #     return result_array
    # else:
    #     return np.array(concentration_data).reshape(-1)