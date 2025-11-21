import numpy as np
import matplotlib.pyplot as plt
import flopy

def transport_model(
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

    delr = Lx / ncol
    delc = Ly / nrow

    top = 10.0
    botm = 0.0
    
    hk = 50.0
    
    nper = 1
    perlen = np.array([perlen])
    nstp = [nstp]

    gwfname = f"gwf_{sim_name}"
    
    sim = flopy.mf6.MFSimulation(
        sim_name=gwfname,
        sim_ws=sim_ws,
        exe_name='mf6',
        verbosity_level=0
    )
    
    flopy.mf6.ModflowTdis(
        sim,
        pname='tdis',
        time_units='DAYS',
        nper=nper,
        perioddata=[(perlen[0], nstp[0], 1.0)]
    )
    
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwfname, save_flows=True)
    
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
    
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname='npf',
        save_flows=True,
        icelltype=0,
        k=hk
    )
    
    flopy.mf6.ModflowGwfsto(gwf, pname='sto', save_flows=True, iconvert=1, ss=0.0, sy=0.0)
    
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

    m=0
    gwt_models = {}
    for species_name, species_initial_conc in species_conc.items():
        nouter, ninner = 50, 100
        hclose, rclose, relax = 1e-6, 1e-6, 0.97
        porosity = 0.3
        alh = 10.0
        ath1 =  alh * 0.3
        atv = alh * 0.1
        diffc = 0.0
        
        gwtname = f"gwt_{species_name}_{sim.name.split('_')[1]}"
        
        gwt = flopy.mf6.ModflowGwt(
            sim, 
            modelname=gwtname, 
            save_flows=True,         
            model_nam_file=f"{gwtname}.nam"
        )
        
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

        sourcerecarray = [("WEL-1", "AUX", species_name),
                          ("chd-1", "AUX", species_name)
                        ]
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
    
    # # 获取结果数据
    # head = gwf.oc.output.head().get_alldata()
    # # print(head.shape)
    # # plt.plot(head[0, 0, 0, :])
    # # plt.show()
    # concentration_data = []
    # species_names = []
    
    # for species, gwt in gwt_models.items():
    #     concentration_data.append(gwt.oc.output.concentration().get_alldata())
    #     species_names.append(species)

    # return head

    # # # 如果输入是一维数组，则将结果转换回一维数组格式返回
    # # if initial_conc is not None and not isinstance(initial_conc, dict):
    # #     # 获取最终的浓度数据
    # #     result_array = np.zeros(len(species_list) * ncol)
        
    # #     # 将每个溶质的浓度数据填充到结果数组中
    # #     for i, species in enumerate(species_list):
    # #         # 获取该溶质的浓度数据
    # #         species_idx = species_names.index(species)
    # #         species_conc_data = concentration_data[species_idx][0][0][0]
            
    # #         # 填充到结果数组中
    # #         result_array[i * ncol:(i + 1) * ncol] = species_conc_data
        
    # #     return result_array
    # # else:
    # #     # 如果输入是字典或None，返回原始格式的结果
    # #     # return concentration_data
    # #     return head