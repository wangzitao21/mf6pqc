from __future__ import annotations

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


def build_transport_model(
    sim_ws: str,
    species_list: list[str],
    initial_conc: np.ndarray,
    boundary_conc: np.ndarray,
    *,
    ncol: int = 16,
    length: float = 6.0,
    porosity: float = 1.0,
    pore_velocity: float = 100.0 / 365.25,
    dispersivity: float = 0.2,
    perlen: float = 0.5 * 365.25,
    nstp: int = 125,
    boundary_node_species: str | None = None,
    advection_scheme: str = "TVD",
) -> None:
    """Build the 1-D inlet-decay transport problem in MODFLOW day units."""
    if boundary_node_species is not None and ncol < 2:
        raise ValueError("An explicit boundary node requires at least two nodes")
    delr = length / (ncol - 1 if boundary_node_species is not None else ncol)
    hydraulic_conductivity = pore_velocity * porosity * length

    simulation = flopy.mf6.MFSimulation(
        sim_name="splitting_decay",
        sim_ws=sim_ws,
        exe_name="./bin/mf6.7.0/mf6.exe",
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        simulation,
        time_units="DAYS",
        nper=1,
        perioddata=[(perlen, nstp, 1.0)],
    )

    gwf = flopy.mf6.ModflowGwf(
        simulation, modelname="gwf_model", save_flows=True
    )
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        complexity="SIMPLE",
        outer_maximum=100,
        outer_dvclose=1.0e-8,
        inner_maximum=200,
        inner_dvclose=1.0e-9,
        rcloserecord=1.0e-9,
        linear_acceleration="CG",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=1,
        ncol=ncol,
        delr=delr,
        delc=1.0,
        top=1.0,
        botm=0.0,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=np.linspace(1.0, 0.0, ncol))
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=hydraulic_conductivity,
        save_specific_discharge=True,
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0)
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="inlet",
        auxiliary=species_list,
        stress_period_data=[((0, 0, 0), 1.0, *boundary_conc)],
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="outlet",
        stress_period_data=[((0, 0, ncol - 1), 0.0)],
    )

    species_conc = {}
    for index, species_name in enumerate(species_list):
        start = index * ncol
        species_conc[species_name] = initial_conc[start : start + ncol]

    src_data = [((0, 0, column), 0.0) for column in range(ncol)]
    for species_name, starting_concentration in species_conc.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation,
            modelname=gwt_name,
            save_flows=True,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            pname=f"{gwt_name}_ims",
            filename=f"{gwt_name}.ims",
            outer_maximum=100,
            outer_dvclose=1.0e-8,
            inner_maximum=200,
            inner_dvclose=1.0e-9,
            rcloserecord=1.0e-9,
            linear_acceleration="BICGSTAB",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=1,
            nrow=1,
            ncol=ncol,
            delr=delr,
            delc=1.0,
            top=1.0,
            botm=0.0,
        )
        flopy.mf6.ModflowGwtic(gwt, strt=starting_concentration)
        flopy.mf6.ModflowGwtadv(gwt, scheme=advection_scheme)
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=dispersivity,
            ath1=0.0,
            diffc=0.0,
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        flopy.mf6.ModflowGwtsrc(
            gwt,
            pname="SRC",
            maxbound=ncol,
            stress_period_data=src_data,
        )
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[("inlet", "AUX", species_name)]
        )
        if species_name == boundary_node_species:
            # The published problem prescribes C(0,t), whereas an AUX value
            # alone is an inflow concentration.  CNC makes the first model
            # cell an explicit x=0 boundary node.  Other transported chemical
            # components retain the chemically consistent AUX inflow.
            flopy.mf6.ModflowGwtcnc(
                gwt,
                pname="CNC",
                maxbound=1,
                stress_period_data=[((0, 0, 0), boundary_conc[index])],
            )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )

    simulation.write_simulation(silent=True)
