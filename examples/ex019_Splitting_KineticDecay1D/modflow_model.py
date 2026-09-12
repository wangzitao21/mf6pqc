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
    ncol: int,
    length: float,
    perlen: float,
    nstp: int,
    porosity: float,
    alh: float,
    advection_scheme: str,
    boundary_node_species: str | None,
    pore_velocity: float,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    inflow_concentrations = boundary_values(species, inflow_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    if boundary_node_species is not None and ncol < 2:
        raise ValueError("An explicit boundary node requires at least two nodes")
    delr = length / (ncol - 1 if boundary_node_species is not None else ncol)
    hydraulic_conductivity = pore_velocity * porosity * length
    simulation = flopy.mf6.MFSimulation(
        sim_name="splitting_decay",
        sim_ws=str(workspace),
        exe_name=mf6_executable,
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)])
    gwf = flopy.mf6.ModflowGwf(simulation, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_maximum=100,
        outer_dvclose=1e-08,
        inner_maximum=200,
        inner_dvclose=1e-09,
        rcloserecord=1e-09,
        linear_acceleration="CG",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=1,
        nrow=1,
        ncol=ncol,
        delr=delr,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfdis(gwf, **discretization)
    flopy.mf6.ModflowGwfic(gwf, strt=np.linspace(1.0, 0.0, ncol))
    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=0, k=hydraulic_conductivity, save_specific_discharge=True
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0)
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="inlet",
        auxiliary=species,
        stress_period_data=[((0, 0, 0), 1.0, *inflow_concentrations)],
    )
    flopy.mf6.ModflowGwfchd(gwf, pname="outlet", stress_period_data=[((0, 0, ncol - 1), 0.0)])
    src_data = [((0, 0, column), 0.0) for column in range(ncol)]
    for species_index, (species_name, concentration) in enumerate(initial_fields.items()):
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=True, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            pname=f"{gwt_name}_ims",
            filename=f"{gwt_name}.ims",
            outer_maximum=100,
            outer_dvclose=1e-08,
            inner_maximum=200,
            inner_dvclose=1e-09,
            rcloserecord=1e-09,
            linear_acceleration="BICGSTAB",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration)
        flopy.mf6.ModflowGwtadv(gwt, scheme=advection_scheme)
        flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=alh, ath1=0.0, diffc=0.0)
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        flopy.mf6.ModflowGwtsrc(gwt, pname="SRC", maxbound=ncol, stress_period_data=src_data)
        flopy.mf6.ModflowGwtssm(gwt, sources=[("inlet", "AUX", species_name)])
        if species_name == boundary_node_species:
            flopy.mf6.ModflowGwtcnc(
                gwt,
                pname="CNC",
                maxbound=1,
                stress_period_data=[((0, 0, 0), inflow_concentrations[species_index])],
            )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )
    simulation.write_simulation(silent=True)
    return simulation
