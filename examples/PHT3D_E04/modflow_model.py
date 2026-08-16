"""MODFLOW 6 flow and transport model for PHT3D Example 4."""

from __future__ import annotations

from pathlib import Path

import flopy


NLAY = 1
NROW = 1
NCOL = 40
NXYZ = NLAY * NROW * NCOL

DELR = 0.002
DELC = 1.0
TOP = 1.0
BOTM = 0.0
POROSITY = 1.0

# The PHT3D input uses seconds. MF6PQC models use days, so all flow and time
# quantities are converted consistently here.
SECONDS_PER_DAY = 86_400.0
SIMULATION_TIME = 20_736.0 / SECONDS_PER_DAY
OUTPUT_INTERVALS = 120
# PHT3D limits the advective Courant number to 0.75. Two equal MF6 coupling
# steps per official output interval give Courant ~= 0.5.
TIME_STEPS = 2 * OUTPUT_INTERVALS
TIME_STEP = SIMULATION_TIME / TIME_STEPS
WELL_RATE = 1.15741e-5 * SECONDS_PER_DAY
# The prescribed well fixes the Darcy flux. A conductivity of 1 m/d is used
# instead of the unit-converted 86,400 m/d to avoid resolving a sub-micrometre
# head gradient; this changes heads only, not the steady transport velocity.
HYDRAULIC_CONDUCTIVITY = 1.0

LONGITUDINAL_DISPERSIVITY = 0.002
TRANSVERSE_DISPERSIVITY = 0.0002


def transport_model(
    *,
    sim_ws: str | Path,
    species_list: list[str],
    initial_conc,
    inflow_concentrations,
    mf6_exe: str | Path,
) -> None:
    """Write the one-dimensional cation-exchange column model."""
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=str(sim_ws),
        exe_name=str(mf6_exe),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=1,
        perioddata=[(SIMULATION_TIME, TIME_STEPS, 1.0)],
    )

    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=1.0e-10,
        outer_maximum=100,
        inner_maximum=200,
        inner_dvclose=1.0e-11,
        rcloserecord=1.0e-10,
        linear_acceleration="CG",
        relaxation_factor=0.97,
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
        k=HYDRAULIC_CONDUCTIVITY,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="INJECTION",
        save_flows=True,
        auxiliary=species_list,
        stress_period_data={
            0: [[(0, 0, 0), WELL_RATE, *inflow_concentrations]]
        },
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="OUTFLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, NCOL - 1), 1.0]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    species_concentrations = {
        name: initial_conc[index * NXYZ : (index + 1) * NXYZ]
        for index, name in enumerate(species_list)
    }

    for species_name, species_initial_conc in species_concentrations.items():
        gwt_name = f"gwt_{species_name}_model"
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=True,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="SUMMARY",
            outer_dvclose=1.0e-9,
            outer_maximum=200,
            inner_maximum=300,
            inner_dvclose=1.0e-10,
            rcloserecord=1.0e-9,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
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
            gwt,
            strt=species_initial_conc,
            filename=f"{gwt_name}.ic",
        )
        flopy.mf6.ModflowGwtadv(
            gwt,
            scheme="TVD",
            filename=f"{gwt_name}.adv",
        )
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=LONGITUDINAL_DISPERSIVITY,
            ath1=TRANSVERSE_DISPERSIVITY,
            atv=TRANSVERSE_DISPERSIVITY,
            diffc=0.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            gwt,
            porosity=POROSITY,
            filename=f"{gwt_name}.mst",
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=[("INJECTION", "AUX", species_name)],
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
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )

    sim.write_simulation(silent=True)
