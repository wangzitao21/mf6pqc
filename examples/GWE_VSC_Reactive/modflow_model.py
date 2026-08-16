"""Small coupled GWF/GWT/GWE/VSC model used by the thermal example."""

from __future__ import annotations

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


NLAY = 1
NROW = 1
NCOL = 41
NXYZ = NLAY * NROW * NCOL
DELR = 0.25
POROSITY = 0.30
BASE_K = 1.0
INITIAL_TEMPERATURE = 20.0
INFLOW_TEMPERATURE = 60.0
SIMULATION_TIME = 20.0
TIME_STEPS = 40
VISCOSITY_REFERENCE = 1.0016e-3


def _ims(sim, model_name: str, filename: str):
    package = flopy.mf6.ModflowIms(
        sim,
        print_option="SUMMARY",
        outer_dvclose=1.0e-9,
        outer_maximum=100,
        inner_maximum=200,
        inner_dvclose=1.0e-10,
        rcloserecord=1.0e-9,
        linear_acceleration="BICGSTAB",
        scaling_method="DIAGONAL",
        reordering_method="RCM",
        relaxation_factor=0.97,
        filename=filename,
    )
    sim.register_ims_package(package, [model_name])


def build_model(
    *,
    sim_ws: str,
    species_list: list[str],
    initial_concentrations: np.ndarray,
    inflow_concentrations: np.ndarray,
    mf6_exe: str,
) -> None:
    """Write one flow model, one GWT per component, and one GWE model."""
    simulation = flopy.mf6.MFSimulation(
        sim_name="gwe_vsc_reactive",
        sim_ws=sim_ws,
        exe_name=mf6_exe,
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        simulation,
        time_units="DAYS",
        nper=1,
        perioddata=[(SIMULATION_TIME, TIME_STEPS, 1.0)],
    )

    flow_name = "gwf_model"
    flow = flopy.mf6.ModflowGwf(
        simulation, modelname=flow_name, save_flows=True
    )
    _ims(simulation, flow.name, f"{flow_name}.ims")
    flopy.mf6.ModflowGwfdis(
        flow,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfic(flow, strt=np.linspace(1.0, 0.0, NCOL))
    flopy.mf6.ModflowGwfnpf(
        flow,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=BASE_K,
        k33=BASE_K,
    )
    flopy.mf6.ModflowGwfvsc(
        flow,
        pname="VSC",
        viscref=VISCOSITY_REFERENCE,
        thermal_formulation="NONLINEAR",
        thermal_a2=10.0,
        thermal_a3=248.37,
        thermal_a4=133.15,
        nviscspecies=1,
        packagedata=[(0, 0.0, INITIAL_TEMPERATURE, "gwe_model", "TEMPERATURE")],
        viscosity_filerecord=f"{flow_name}.vsc.bin",
    )
    flopy.mf6.ModflowGwfsto(
        flow,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )
    flopy.mf6.ModflowGwfchd(
        flow,
        pname="CHD-FLOW",
        save_flows=True,
        stress_period_data={
            0: [[(0, 0, 0), 1.0], [(0, 0, NCOL - 1), 0.0]]
        },
    )
    flopy.mf6.ModflowGwfoc(
        flow,
        budget_filerecord=f"{flow_name}.bud",
        head_filerecord=f"{flow_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )

    for component_index, component in enumerate(species_list):
        model_name = get_gwt_model_name(component)
        transport = flopy.mf6.ModflowGwt(
            simulation,
            modelname=model_name,
            save_flows=True,
            model_nam_file=f"{model_name}.nam",
        )
        _ims(simulation, transport.name, f"{model_name}.ims")
        flopy.mf6.ModflowGwtdis(
            transport,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=1.0,
            top=1.0,
            botm=0.0,
        )
        start = component_index * NXYZ
        stop = start + NXYZ
        flopy.mf6.ModflowGwtic(
            transport, strt=initial_concentrations[start:stop]
        )
        flopy.mf6.ModflowGwtadv(transport, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(
            transport,
            xt3d_off=True,
            alh=0.05,
            ath1=0.005,
            atv=0.005,
            diffc=1.0e-5,
        )
        flopy.mf6.ModflowGwtmst(transport, porosity=POROSITY)
        flopy.mf6.ModflowGwtcnc(
            transport,
            pname="CNC-INFLOW",
            stress_period_data={
                0: [[(0, 0, 0), float(inflow_concentrations[component_index])]]
            },
        )
        flopy.mf6.ModflowGwtssm(transport)
        flopy.mf6.ModflowGwtoc(
            transport,
            budget_filerecord=f"{model_name}.cbc",
            concentration_filerecord=f"{model_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=flow_name,
            exgmnameb=model_name,
            filename=f"{model_name}.gwfgwt",
        )

    energy_name = "gwe_model"
    energy = flopy.mf6.ModflowGwe(
        simulation,
        modelname=energy_name,
        save_flows=True,
        model_nam_file=f"{energy_name}.nam",
    )
    _ims(simulation, energy.name, f"{energy_name}.ims")
    flopy.mf6.ModflowGwedis(
        energy,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGweic(energy, strt=INITIAL_TEMPERATURE)
    flopy.mf6.ModflowGweadv(energy, scheme="TVD")
    flopy.mf6.ModflowGweest(
        energy,
        pname="EST",
        save_flows=True,
        porosity=POROSITY,
        density_water=1000.0,
        heat_capacity_water=4184.0,
        density_solid=2650.0,
        heat_capacity_solid=800.0,
    )
    # W/(m K) is converted to J/(day m K) because TDIS uses days.
    flopy.mf6.ModflowGwecnd(
        energy,
        xt3d_off=True,
        alh=0.05,
        ath1=0.005,
        atv=0.005,
        ktw=0.60 * 86_400.0,
        kts=2.00 * 86_400.0,
    )
    flopy.mf6.ModflowGwectp(
        energy,
        pname="CTP-TEMPERATURE",
        stress_period_data={
            0: [
                [(0, 0, 0), INFLOW_TEMPERATURE],
                [(0, 0, NCOL - 1), INITIAL_TEMPERATURE],
            ]
        },
    )
    flopy.mf6.ModflowGwessm(energy)
    flopy.mf6.ModflowGweoc(
        energy,
        budget_filerecord=f"{energy_name}.cbc",
        temperature_filerecord=f"{energy_name}.ucn",
        saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "LAST")],
    )
    flopy.mf6.ModflowGwfgwe(
        simulation,
        exgtype="GWF6-GWE6",
        exgmnamea=flow_name,
        exgmnameb=energy_name,
        filename=f"{energy_name}.gwfgwe",
    )
    simulation.write_simulation(silent=True)
