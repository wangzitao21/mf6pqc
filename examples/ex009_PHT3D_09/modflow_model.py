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
    background_concentrations: np.ndarray,
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
    hydraulic_conductivity: float,
    inlet_head: float,
    outlet_head: float,
    inflow_rate: float,
    well_cell: tuple[int, int, int],
    alh: float,
    ath1: float,
    atv: float,
    diffc: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nlay * nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    background_concentrations = boundary_values(species, background_concentrations)
    inflow_concentrations = boundary_values(species, inflow_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    sim_name = "model"
    gwf_name = f"gwf_{sim_name}"

    simulation = flopy.mf6.MFSimulation(
        sim_name=sim_name, sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, pname="tdis", time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)]
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_dvclose=1e-08,
        outer_maximum=50,
        inner_maximum=100,
        inner_dvclose=1e-09,
        rcloserecord=1e-06,
        linear_acceleration="CG",
        relaxation_factor=0.97,
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=length_x / ncol,
        delc=length_y / nrow,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfdis(gwf, pname="dis", **discretization)
    initial_head = np.full((nlay, nrow, ncol), outlet_head, dtype=float)
    initial_head[:, :, 0] = inlet_head
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfsto(gwf, pname="sto", iconvert=1, ss=0.0, sy=0.0, steady_state={0: True})
    left_chd = [[(0, row, 0), inlet_head, *background_concentrations] for row in range(nrow)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-LEFT",
        save_flows=True,
        stress_period_data={0: left_chd},
        auxiliary=species,
        filename=f"{gwf_name}.left.chd",
    )
    right_chd = [[(0, row, ncol - 1), outlet_head] for row in range(nrow)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-RIGHT",
        save_flows=True,
        stress_period_data={0: right_chd},
        filename=f"{gwf_name}.right.chd",
    )
    well_data = [[well_cell, inflow_rate, *inflow_concentrations]]
    flopy.mf6.ModflowGwfwel(
        gwf, pname="WEL-1", save_flows=True, stress_period_data={0: well_data}, auxiliary=species
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=True, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="SUMMARY",
            outer_dvclose=1e-06,
            outer_maximum=50,
            inner_maximum=100,
            inner_dvclose=1e-06,
            rcloserecord=1e-06,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt, xt3d_off=True, alh=alh, ath1=ath1, atv=atv, diffc=diffc, filename=f"{gwt_name}.dsp"
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=[("WEL-1", "AUX", species_name), ("CHD-LEFT", "AUX", species_name)],
            filename=f"{gwt_name}.ssm",
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
