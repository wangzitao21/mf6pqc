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
    ambient_concentrations: np.ndarray,
    recharge_concentrations: np.ndarray,
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
    outlet_head: float,
    recharge_rate: float,
    alh: float,
    alv: float,
    ath1: float,
    ath2: float,
    atv: float,
    diffc: float,
    boundary_conductivity: float,
    first_boundary_layer: int,
    left_rates: float | list[float] | np.ndarray,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nlay * nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    ambient_concentrations = boundary_values(species, ambient_concentrations)
    recharge_concentrations = boundary_values(species, recharge_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    if np.ndim(delr) > 0 and np.size(delr) != ncol:
        raise ValueError("Cell widths must be a scalar or match ncol")
    if np.ndim(botm) > 0 and np.size(botm) != nlay:
        raise ValueError("Layer bottoms must be a scalar or match nlay")
    left_rates = np.broadcast_to(
        np.asarray(left_rates, dtype=float), (nlay - first_boundary_layer,)
    )
    sim_name = "model"
    gwf_name = f"gwf_{sim_name}"
    simulation = flopy.mf6.MFSimulation(
        sim_name=sim_name, sim_ws=str(workspace), exe_name=mf6_executable, verbosity_level=0
    )
    flopy.mf6.ModflowTdis(
        simulation, pname="tdis", time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)]
    )
    gwf = flopy.mf6.ModflowGwf(
        simulation, modelname=gwf_name, save_flows=True, newtonoptions="NEWTON"
    )
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        print_option="SUMMARY",
        complexity="MODERATE",
        outer_dvclose=1e-09,
        outer_maximum=200,
        inner_maximum=200,
        inner_dvclose=1e-10,
        rcloserecord=1e-08,
        linear_acceleration="BICGSTAB",
        relaxation_factor=0.97,
        filename="flow.ims",
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
    flopy.mf6.ModflowGwfic(
        gwf, pname="ic", strt=np.full((nlay, nrow, ncol), outlet_head, dtype=float)
    )
    vertical_conductivity = np.full((nlay, nrow, ncol), hydraulic_conductivity, dtype=float)
    vertical_conductivity[:, 0, 0] = boundary_conductivity
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=1,
        k=hydraulic_conductivity,
        k33=vertical_conductivity,
        wetdry=-0.01,
    )
    flopy.mf6.ModflowGwfsto(gwf, pname="sto", iconvert=1, ss=0.0, sy=0.0, steady_state={0: True})

    well_concentrations = np.zeros(len(species), dtype=float)
    left_wells = [
        [(layer, 0, 0), float(rate), *well_concentrations]
        for layer, rate in zip(range(first_boundary_layer, nlay), left_rates, strict=True)
    ]
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-LEFT",
        save_flows=True,
        maxbound=len(left_wells),
        stress_period_data={0: left_wells},
        auxiliary=species,
    )
    right_chd = [[(layer, 0, ncol - 1), outlet_head] for layer in range(first_boundary_layer, nlay)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-RIGHT",
        save_flows=True,
        maxbound=len(right_chd),
        stress_period_data={0: right_chd},
    )
    recharge = [[(0, 0, col), recharge_rate, *recharge_concentrations] for col in range(ncol)]
    flopy.mf6.ModflowGwfrch(
        gwf,
        pname="RCH-TOP",
        fixed_cell=False,
        save_flows=True,
        maxbound=len(recharge),
        stress_period_data={0: recharge},
        auxiliary=species,
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )
    for species_index, (species_name, concentration) in enumerate(initial_fields.items()):
        gwt_name = get_gwt_model_name(species_name)
        is_mobile = species_name != "Charge"
        needs_isotope_precision = species_name in {"Tolu_h", "Naph_h"}
        transport_outer_dvclose = 1e-09 if needs_isotope_precision else 1e-07
        transport_inner_dvclose = 1e-12 if needs_isotope_precision else 1e-08
        transport_rclose = 1e-12 if needs_isotope_precision else 1e-07
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=False, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            print_option="NONE",
            complexity="MODERATE",
            outer_dvclose=transport_outer_dvclose,
            outer_maximum=200,
            inner_maximum=1000,
            inner_dvclose=transport_inner_dvclose,
            rcloserecord=transport_rclose,
            linear_acceleration="BICGSTAB",
            relaxation_factor=0.97,
            filename=f"{gwt_name}.ims",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, filename=f"{gwt_name}.dis", **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration, filename=f"{gwt_name}.ic")
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity, filename=f"{gwt_name}.mst")
        if is_mobile:
            flopy.mf6.ModflowGwtadv(gwt, scheme="TVD", filename=f"{gwt_name}.adv")
            flopy.mf6.ModflowGwtdsp(
                gwt,
                alh=alh,
                alv=alv,
                ath1=ath1,
                ath2=ath2,
                atv=atv,
                diffc=diffc,
                filename=f"{gwt_name}.dsp",
            )
            flopy.mf6.ModflowGwtssm(
                gwt,
                sources=[("WEL-LEFT", "AUX", species_name), ("RCH-TOP", "AUX", species_name)],
                filename=f"{gwt_name}.ssm",
            )
            cnc_data = [
                [(layer, 0, col), float(ambient_concentrations[species_index])]
                for layer in range(nlay)
                for col in range(6)
            ]
        else:
            cnc_data = [[(layer, 0, col), 0.0] for layer in range(nlay) for col in range(ncol)]
        flopy.mf6.ModflowGwtcnc(
            gwt,
            pname="CNC-UPSTREAM" if is_mobile else "CNC-ZERO-CHARGE",
            maxbound=len(cnc_data),
            stress_period_data={0: cnc_data},
            filename=f"{gwt_name}.cnc",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        if is_mobile:
            flopy.mf6.ModflowGwfgwt(
                simulation,
                exgtype="GWF6-GWT6",
                exgmnamea=gwf_name,
                exgmnameb=gwt_name,
                filename=f"{gwt_name}.gwfgwt",
            )
    simulation.write_simulation(silent=True)
    return simulation
