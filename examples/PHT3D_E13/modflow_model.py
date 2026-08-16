"""MODFLOW 6 model for PHT3D Example 13."""

from __future__ import annotations

from pathlib import Path

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


NLAY = 1
NROW = 1
NCOL = 16
NXYZ = NLAY * NROW * NCOL
DELR = 0.0033125
DELC = 1.0
TOP = 0.00287433
BOTM = 0.0
POROSITY = 0.376
FLOW_RATE = 2.4e-4
PERIOD_DATA = [(0.9333333, 64, 1.0), (1.458333, 100, 1.0)]


def _component_fields(
    species_list: list[str], initial_concentrations: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: initial_concentrations[index * NXYZ : (index + 1) * NXYZ]
        for index, name in enumerate(species_list)
    }


def transport_model(
    *,
    sim_ws: str | Path,
    species_list: list[str],
    initial_conc: np.ndarray,
    pulse_concentrations: np.ndarray,
    chase_concentrations: np.ndarray,
    mf6_exe: str | Path,
) -> None:
    auxiliary_names = [f"SP{index:03d}" for index in range(len(species_list))]
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=str(sim_ws),
        exe_name=str(mf6_exe),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim, time_units="DAYS", nper=2, perioddata=PERIOD_DATA
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_dvclose=1.0e-10,
        inner_dvclose=1.0e-11,
        rcloserecord=1.0e-10,
        linear_acceleration="CG",
        filename="flow.ims",
    )
    sim.register_ims_package(flow_ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=DELC,
        top=TOP,
        botm=BOTM,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=1.0)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=1.0,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True, 1: True},
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-INLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={
            0: [[(0, 0, 0), FLOW_RATE, *pulse_concentrations]],
            1: [[(0, 0, 0), FLOW_RATE, *chase_concentrations]],
        },
        auxiliary=auxiliary_names,
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-OUTLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={0: [[(0, 0, NCOL - 1), 1.0]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="gwf_model.bud",
        head_filerecord="gwf_model.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    for species_index, (species, concentration) in enumerate(
        _component_fields(species_list, initial_conc).items()
    ):
        gwt_name = get_gwt_model_name(species)
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=False,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            complexity="SIMPLE",
            outer_dvclose=1.0e-8,
            outer_maximum=100,
            inner_maximum=100,
            inner_dvclose=1.0e-9,
            rcloserecord=1.0e-8,
            linear_acceleration="BICGSTAB",
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            filename=f"{gwt_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            gwt, strt=concentration, filename=f"{gwt_name}.ic"
        )
        flopy.mf6.ModflowGwtadv(
            gwt, scheme="TVD", filename=f"{gwt_name}.adv"
        )
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=0.00537,
            ath1=0.000537,
            diffc=0.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            gwt, porosity=POROSITY, filename=f"{gwt_name}.mst"
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=[
                ("WEL-INLET", "AUX", auxiliary_names[species_index])
            ],
            filename=f"{gwt_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )

    sim.write_simulation(silent=True)
