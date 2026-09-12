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
    period_data: list[tuple[float, int, float]],
    porosity: float,
    hydraulic_conductivity: float | np.ndarray,
    vertical_conductivity_ratio: float,
    initial_head: float | np.ndarray | None,
    inlet_head: float,
    outlet_head: float,
    alh: float,
    ath1: float,
    diffc: float,
    boundary_distance: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nlay * nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    if nlay != 1 or nrow != 1:
        raise ValueError("This example requires one layer and one row")
    inflow_concentrations = boundary_values(species, inflow_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    gwf_name = "gwf_model"
    if np.ndim(delr) > 0 and np.size(delr) != ncol:
        raise ValueError("Cell widths must be a scalar or match ncol")

    nper = len(period_data)

    conductivity = np.broadcast_to(hydraulic_conductivity, (nlay, nrow, ncol))
    left_conductance = float(conductivity[0, 0, 0]) / boundary_distance
    right_conductance = float(conductivity[0, 0, -1]) / boundary_distance
    simulation = flopy.mf6.MFSimulation(
        sim_name="model", sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, pname="tdis", time_units="DAYS", nper=nper, perioddata=period_data
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="ims",
        complexity="SIMPLE",
        outer_dvclose=1e-08,
        outer_maximum=50,
        under_relaxation="NONE",
        inner_maximum=500,
        inner_dvclose=1e-09,
        rcloserecord=1e-08,
        linear_acceleration="BICGSTAB",
        scaling_method="DIAGONAL",
        reordering_method="RCM",
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
        save_flows=False,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity * vertical_conductivity_ratio,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=initial_head)
    flopy.mf6.ModflowGwfsto(gwf, pname="sto", save_flows=False, iconvert=1, ss=0.0, sy=0.0)
    ghb_spd = [[(0, 0, ncol - 1), outlet_head, right_conductance]]
    flopy.mf6.ModflowGwfghb(
        gwf,
        pname="ghb_right",
        save_flows=True,
        maxbound=len(ghb_spd),
        stress_period_data={0: ghb_spd},
        filename=f"{gwf_name}.choushui.ghb",
    )
    ghb2_spd = [[(0, 0, 0), inlet_head, left_conductance, *inflow_concentrations]]
    flopy.mf6.ModflowGwfghb(
        gwf,
        pname="bushui",
        save_flows=True,
        maxbound=len(ghb2_spd),
        stress_period_data={0: ghb2_spd},
        auxiliary=species,
        filename=f"{gwf_name}.bushui.ghb",
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")],
    )
    nouter, ninner = (50, 100)
    hclose, rclose, relax = (1e-06, 1e-06, 1.0)

    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=False, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
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
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, idomain=1, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtadv(gwt, scheme="UPSTREAM", filename=f"{gwt_name}.adv")
        flopy.mf6.ModflowGwtdsp(
            gwt, xt3d_off=True, alh=alh, ath1=ath1, diffc=diffc, filename=f"{gwt_name}.dsp"
        )
        flopy.mf6.ModflowGwtmst(gwt, pname="mst", porosity=porosity, filename=f"{gwt_name}.mst")
        sourcerecarray = [("bushui", "AUX", species_name)]
        flopy.mf6.ModflowGwtssm(
            gwt, pname=f"{species_name}_ssm", sources=sourcerecarray, filename=f"{gwt_name}.ssm"
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            budgetcsv_filerecord=f"{gwt_name}.oc.csv",
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
