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
    delr: float | list[float] | np.ndarray,
    delc: float | list[float] | np.ndarray,
    top: float | list[float] | np.ndarray,
    botm: float | list[float] | np.ndarray,
    perlen: float,
    nstp: int,
    porosity: float,
    hydraulic_conductivity: float,
    initial_head: float,
    outlet_head: float,
    inflow_rate: float,
    alh: float,
    ath1: float,
    diffc: float,
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
    simulation = flopy.mf6.MFSimulation(
        sim_name="model", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)])
    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        outer_dvclose=1e-10,
        outer_maximum=100,
        inner_maximum=200,
        inner_dvclose=1e-11,
        rcloserecord=1e-10,
        linear_acceleration="CG",
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
    flopy.mf6.ModflowGwfdis(gwf, **discretization)
    flopy.mf6.ModflowGwfic(gwf, strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf, save_flows=True, save_specific_discharge=True, icelltype=0, k=hydraulic_conductivity
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0, steady_state={0: True})
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="INJECTION",
        save_flows=True,
        auxiliary=species,
        stress_period_data={0: [[(0, 0, 0), inflow_rate, *inflow_concentrations]]},
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="OUTFLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, ncol - 1), outlet_head]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
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
            outer_dvclose=1e-09,
            outer_maximum=200,
            inner_maximum=300,
            inner_dvclose=1e-10,
            rcloserecord=1e-09,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=alh,
            ath1=ath1,
            atv=ath1,
            diffc=diffc,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[("INJECTION", "AUX", species_name)], filename=f"{gwt_name}.ssm"
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
