from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import flopy
import numpy as np
from example_utils import boundary_values, component_fields, executable_path

from mf6pqc.utils import get_gwt_model_name


def build_model(
    *,
    workspace: str | Path,
    species: list[str],
    initial_concentrations: np.ndarray,
    inflow_concentrations: np.ndarray,
    nlay: int,
    nrow: int,
    ncol: int,
    length_x: float,
    length_y: float,
    top: float | list[float] | np.ndarray,
    botm: float | list[float] | np.ndarray,
    perlen: float,
    nstp: int,
    porosity: float,
    hydraulic_conductivity: np.ndarray | None,
    initial_head: float | np.ndarray | None,
    inlet_head: float,
    outlet_head: float,
    alh: float,
    ath1: float,
    diffc: float,
    tsmult: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nlay * nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    inflow_concentrations = boundary_values(species, inflow_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    sim_name = "model"
    gwf_name = f"gwf_{sim_name}"

    delr = length_x / ncol
    delc = length_y / nrow

    nper = 1

    simulation = flopy.mf6.MFSimulation(
        sim_name=sim_name, sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, pname="tdis", time_units="DAYS", nper=nper, perioddata=[(perlen, nstp, tsmult)]
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=False)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="ims",
        complexity="SIMPLE",
        outer_dvclose=1e-08,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=100,
        inner_dvclose=1e-09,
        rcloserecord=1e-10,
        linear_acceleration="CG",
        scaling_method="NONE",
        reordering_method="NONE",
        relaxation_factor=0.97,
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfdis(gwf, pname="dis", **discretization)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=hydraulic_conductivity,
    )
    if initial_head is None:
        initial_head = np.full((nlay, nrow, ncol), inlet_head)
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_head)
    flopy.mf6.ModflowGwfsto(
        gwf, pname="sto", save_flows=False, iconvert=1, ss=0.0, sy=0.0, steady_state={0: True}
    )
    chd_spd_1 = []
    for i in range(nrow):
        chd_spd_1.append([(0, i, 0), inlet_head, *inflow_concentrations])
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd-1",
        save_flows=False,
        maxbound=len(chd_spd_1),
        stress_period_data={0: chd_spd_1},
        auxiliary=species,
        filename=f"{gwf_name}.1.chd",
    )
    chd_spd_2 = []
    for i in range(nrow):
        chd_spd_2.append([(0, i, ncol - 1), outlet_head])
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd-2",
        save_flows=False,
        maxbound=len(chd_spd_2),
        stress_period_data={0: chd_spd_2},
        filename=f"{gwf_name}.2.chd",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=False, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="SUMMARY",
            outer_dvclose=1e-08,
            outer_maximum=100,
            under_relaxation="NONE",
            inner_maximum=300,
            inner_dvclose=1e-10,
            rcloserecord=1e-08,
            linear_acceleration="BICGSTAB",
            scaling_method="NONE",
            reordering_method="NONE",
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, idomain=1, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt, xt3d_off=True, alh=alh, ath1=ath1, diffc=diffc, filename=f"{gwt_name}.dsp"
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        sourcerecarray = [("chd-1", "AUX", species_name)]
        flopy.mf6.ModflowGwtssm(
            gwt, pname=f"{species_name}_ssm", sources=sourcerecarray, filename=f"{gwt_name}.ssm"
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf_name,
            exgmnameb=gwt_name,
            filename=f"{gwt_name}.gwfgwt",
        )
    simulation.write_simulation(silent=True)
    return simulation
