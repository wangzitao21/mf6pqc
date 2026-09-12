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


def _ims(simulation: flopy.mf6.MFSimulation, model_name: str, filename: str) -> None:
    package = flopy.mf6.ModflowIms(
        simulation,
        print_option="SUMMARY",
        outer_dvclose=1e-09,
        outer_maximum=100,
        inner_maximum=200,
        inner_dvclose=1e-10,
        rcloserecord=1e-09,
        linear_acceleration="BICGSTAB",
        scaling_method="DIAGONAL",
        reordering_method="RCM",
        relaxation_factor=0.97,
        filename=filename,
    )
    simulation.register_ims_package(package, [model_name])


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
    inlet_head: float,
    outlet_head: float,
    alh: float,
    ath1: float,
    atv: float,
    diffc: float,
    density_solid: float,
    density_water: float,
    heat_capacity_solid: float,
    heat_capacity_water: float,
    inflow_temperature: float,
    initial_temperature: float,
    kts: float,
    ktw: float,
    thermal_a2: float,
    thermal_a3: float,
    thermal_a4: float,
    viscosity_reference: float,
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
        sim_name="gwe_vsc_reactive",
        sim_ws=str(workspace),
        exe_name=mf6_executable,
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1, perioddata=[(perlen, nstp, 1.0)])
    gwf_name = "gwf_model"
    gwf = flopy.mf6.ModflowGwf(simulation, modelname=gwf_name, save_flows=True)
    _ims(simulation, gwf.name, f"{gwf_name}.ims")
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
    flopy.mf6.ModflowGwfic(gwf, strt=np.linspace(inlet_head, outlet_head, ncol))
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="NPF",
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=hydraulic_conductivity,
        k33=hydraulic_conductivity,
    )
    flopy.mf6.ModflowGwfvsc(
        gwf,
        pname="VSC",
        viscref=viscosity_reference,
        thermal_formulation="NONLINEAR",
        thermal_a2=thermal_a2,
        thermal_a3=thermal_a3,
        thermal_a4=thermal_a4,
        nviscspecies=1,
        packagedata=[(0, 0.0, initial_temperature, "gwe_model", "TEMPERATURE")],
        viscosity_filerecord=f"{gwf_name}.vsc.bin",
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0, steady_state={0: True})
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-FLOW",
        save_flows=True,
        stress_period_data={0: [[(0, 0, 0), inlet_head], [(0, 0, ncol - 1), outlet_head]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord=f"{gwf_name}.bud",
        head_filerecord=f"{gwf_name}.hds",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    )
    for species_index, (species_name, concentration) in enumerate(initial_fields.items()):
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=True, model_nam_file=f"{gwt_name}.nam"
        )
        _ims(simulation, gwt.name, f"{gwt_name}.ims")
        flopy.mf6.ModflowGwtdis(gwt, **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration)
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=alh, ath1=ath1, atv=atv, diffc=diffc)
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        flopy.mf6.ModflowGwtcnc(
            gwt,
            pname="CNC-INFLOW",
            stress_period_data={0: [[(0, 0, 0), float(inflow_concentrations[species_index])]]},
        )
        flopy.mf6.ModflowGwtssm(gwt)
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
    energy_name = "gwe_model"
    energy = flopy.mf6.ModflowGwe(
        simulation, modelname=energy_name, save_flows=True, model_nam_file=f"{energy_name}.nam"
    )
    _ims(simulation, energy.name, f"{energy_name}.ims")
    flopy.mf6.ModflowGwedis(energy, **discretization)
    flopy.mf6.ModflowGweic(energy, strt=initial_temperature)
    flopy.mf6.ModflowGweadv(energy, scheme="TVD")
    flopy.mf6.ModflowGweest(
        energy,
        pname="EST",
        save_flows=True,
        porosity=porosity,
        density_water=density_water,
        heat_capacity_water=heat_capacity_water,
        density_solid=density_solid,
        heat_capacity_solid=heat_capacity_solid,
    )
    flopy.mf6.ModflowGwecnd(energy, xt3d_off=True, alh=alh, ath1=ath1, atv=atv, ktw=ktw, kts=kts)
    flopy.mf6.ModflowGwectp(
        energy,
        pname="CTP-TEMPERATURE",
        stress_period_data={
            0: [[(0, 0, 0), inflow_temperature], [(0, 0, ncol - 1), initial_temperature]]
        },
    )
    flopy.mf6.ModflowGwessm(energy)
    flopy.mf6.ModflowGweoc(
        energy,
        budget_filerecord=f"{energy_name}.cbc",
        temperature_filerecord=f"{energy_name}.ucn",
        saverecord=[("TEMPERATURE", "ALL"), ("BUDGET", "LAST")],
    )
    flopy.mf6.ModflowGwfgwe(
        simulation,
        exgtype="GWF6-GWE6",
        exgmnamea=gwf_name,
        exgmnameb=energy_name,
        filename=f"{energy_name}.gwfgwe",
    )
    simulation.write_simulation(silent=True)
    return simulation
