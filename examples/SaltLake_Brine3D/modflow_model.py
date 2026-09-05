"""MODFLOW 6 GWF/GWT model for the three-dimensional salt-lake case."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name

from case_config import (
    CHANNEL_STAGE_M,
    INITIAL_HEAD_M,
    K33_RATIO,
    SPECIFIC_STORAGE_PER_M,
    CaseProfile,
)


FLOW_MODEL_NAME = "gwf_model"
NPF_PACKAGE_NAME = "NPF"
CHANNEL_PACKAGE_NAME = "RECHARGE_CHANNEL"
WELL_PACKAGE_NAME = "PRODUCTION_WELLS"


def _output_records(profile: CaseProfile, variable: str) -> dict[int, list[tuple]]:
    return {0: [(variable, "STEPS", *profile.save_steps)]}


def build_model(
    *,
    workspace: str | Path,
    mf6_executable: str | Path,
    profile: CaseProfile,
    species: Iterable[str],
    initial_concentrations: np.ndarray,
    channel_concentrations: np.ndarray,
    porosity: np.ndarray,
    hydraulic_conductivity: np.ndarray,
    density_feedback: bool = True,
) -> flopy.mf6.MFSimulation:
    """Build and write the confined GWF plus one GWT model per component."""
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    species = list(species)
    initial_concentrations = np.asarray(initial_concentrations, dtype=float).ravel()
    channel_concentrations = np.asarray(channel_concentrations, dtype=float).ravel()
    porosity = np.asarray(porosity, dtype=float).reshape(profile.shape)
    hydraulic_conductivity = np.asarray(
        hydraulic_conductivity, dtype=float
    ).reshape(profile.shape)

    expected = len(species) * profile.nxyz
    if initial_concentrations.size != expected:
        raise ValueError(
            "Initial concentration vector has size "
            f"{initial_concentrations.size}; expected {expected}"
        )
    if channel_concentrations.size != len(species):
        raise ValueError("Channel concentration vector does not match components")
    if np.any(hydraulic_conductivity <= 0.0):
        raise ValueError("Hydraulic conductivity must be positive")

    simulation = flopy.mf6.MFSimulation(
        sim_name="salt_lake_brine_3d",
        sim_ws=str(workspace),
        exe_name=str(mf6_executable),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        simulation,
        pname="TDIS",
        time_units="DAYS",
        nper=1,
        perioddata=[(profile.period_days, profile.total_steps, 1.0)],
    )

    flow = flopy.mf6.ModflowGwf(
        simulation,
        modelname=FLOW_MODEL_NAME,
        save_flows=True,
    )
    flow_solver = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=1.0e-6,
        outer_maximum=100,
        under_relaxation="NONE",
        inner_maximum=500,
        inner_dvclose=1.0e-8,
        rcloserecord=1.0e-6,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.99,
        filename="flow.ims",
    )
    simulation.register_ims_package(flow_solver, [flow.name])

    flopy.mf6.ModflowGwfdis(
        flow,
        pname="DIS",
        nlay=profile.nlay,
        nrow=profile.nrow,
        ncol=profile.ncol,
        delr=profile.delr,
        delc=profile.delc,
        top=profile.top,
        botm=profile.botm,
    )
    flopy.mf6.ModflowGwfic(flow, pname="IC", strt=INITIAL_HEAD_M)
    flopy.mf6.ModflowGwfnpf(
        flow,
        pname=NPF_PACKAGE_NAME,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity * K33_RATIO,
    )
    flopy.mf6.ModflowGwfsto(
        flow,
        pname="STO",
        save_flows=True,
        iconvert=0,
        ss=SPECIFIC_STORAGE_PER_M,
        sy=0.0,
        transient={0: True},
    )

    channel_data = [
        (
            cell,
            CHANNEL_STAGE_M,
            profile.channel_conductance_per_cell_m2_per_day,
            *channel_concentrations,
        )
        for cell in profile.channel_cells
    ]
    flopy.mf6.ModflowGwfghb(
        flow,
        pname=CHANNEL_PACKAGE_NAME,
        filename="gwf_model.recharge_channel.ghb",
        save_flows=True,
        auxiliary=species,
        stress_period_data={0: channel_data},
    )

    well_data = [
        (cell, profile.well_rate_m3_per_day) for cell in profile.well_cells
    ]
    flopy.mf6.ModflowGwfwel(
        flow,
        pname=WELL_PACKAGE_NAME,
        filename="gwf_model.production_wells.wel",
        save_flows=True,
        stress_period_data={0: well_data},
    )

    flopy.mf6.ModflowGwfoc(
        flow,
        pname="OC",
        budget_filerecord=f"{FLOW_MODEL_NAME}.bud",
        head_filerecord=f"{FLOW_MODEL_NAME}.hds",
        saverecord={
            0: [
                ("HEAD", "STEPS", *profile.save_steps),
                ("BUDGET", "STEPS", *profile.save_steps),
            ]
        },
    )

    component_fields = {
        name: initial_concentrations[
            index * profile.nxyz : (index + 1) * profile.nxyz
        ].reshape(profile.shape)
        for index, name in enumerate(species)
    }

    for species_name, initial_field in component_fields.items():
        transport_name = get_gwt_model_name(species_name)
        transport = flopy.mf6.ModflowGwt(
            simulation,
            modelname=transport_name,
            model_nam_file=f"{transport_name}.nam",
            save_flows=True,
        )
        transport_solver = flopy.mf6.ModflowIms(
            simulation,
            print_option="SUMMARY",
            outer_dvclose=1.0e-7,
            outer_maximum=120,
            inner_maximum=250,
            inner_dvclose=1.0e-8,
            rcloserecord=1.0e-6,
            linear_acceleration="BICGSTAB",
            scaling_method="DIAGONAL",
            reordering_method="RCM",
            relaxation_factor=0.97,
            filename=f"{transport_name}.ims",
        )
        simulation.register_ims_package(transport_solver, [transport.name])

        flopy.mf6.ModflowGwtdis(
            transport,
            pname="DIS",
            nlay=profile.nlay,
            nrow=profile.nrow,
            ncol=profile.ncol,
            delr=profile.delr,
            delc=profile.delc,
            top=profile.top,
            botm=profile.botm,
            filename=f"{transport_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            transport,
            pname="IC",
            strt=initial_field,
            filename=f"{transport_name}.ic",
        )
        flopy.mf6.ModflowGwtadv(
            transport,
            pname="ADV",
            scheme="TVD",
            filename=f"{transport_name}.adv",
        )
        flopy.mf6.ModflowGwtdsp(
            transport,
            pname="DSP",
            xt3d_off=False,
            alh=20.0,
            alv=5.0,
            ath1=2.0,
            atv=0.5,
            diffc=1.0e-9 * 86_400.0,
            filename=f"{transport_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            transport,
            pname="MST",
            porosity=porosity,
            filename=f"{transport_name}.mst",
        )
        flopy.mf6.ModflowGwtssm(
            transport,
            pname=f"{species_name}_SSM",
            sources=[(CHANNEL_PACKAGE_NAME, "AUX", species_name)],
            filename=f"{transport_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            transport,
            pname="OC",
            budget_filerecord=f"{transport_name}.cbc",
            concentration_filerecord=f"{transport_name}.ucn",
            saverecord={
                0: [
                    ("CONCENTRATION", "STEPS", *profile.save_steps),
                    ("BUDGET", "STEPS", *profile.save_steps),
                ]
            },
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=FLOW_MODEL_NAME,
            exgmnameb=transport_name,
            filename=f"{transport_name}.gwfgwt",
        )

    if density_feedback:
        chloride_model = get_gwt_model_name("Cl")
        if "Cl" not in species:
            raise ValueError("Density feedback requires the transported Cl component")
        # A zero-slope BUY entry allocates DENSE. MF6PQC then writes the full
        # Pitzer-calculated density field directly before each flow solve.
        flopy.mf6.ModflowGwfbuy(
            flow,
            pname="BUY",
            denseref=1000.0,
            nrhospecies=1,
            density_filerecord="model_density.bin",
            packagedata=[
                (0, 0.0, 0.0, chloride_model, "CONCENTRATION")
            ],
        )

    simulation.write_simulation(silent=True)
    return simulation
