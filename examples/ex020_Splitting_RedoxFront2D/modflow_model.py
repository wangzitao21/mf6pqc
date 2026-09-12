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
    pulse_concentrations: np.ndarray,
    nrow: int,
    ncol: int,
    length: float,
    width: float,
    top: float | list[float] | np.ndarray,
    botm: float | list[float] | np.ndarray,
    porosity: float,
    hydraulic_conductivity: np.ndarray,
    inlet_head: float,
    outlet_head: float,
    alh: float,
    ath1: float,
    diffc: float,
    flush_duration: float,
    logical_steps_per_period: tuple[int, int],
    pulse_duration: float,
    strang_half_steps: bool,
    mf6_executable: str | Path | None = None,
) -> flopy.mf6.MFSimulation:
    """Build and write the MODFLOW 6 inputs and return the unrun simulation."""
    workspace = Path(workspace).expanduser().resolve()
    species = list(species)
    nxyz = nrow * ncol
    initial_fields = component_fields(species, initial_concentrations, nxyz)
    pulse_concentrations = boundary_values(species, pulse_concentrations)
    background_concentrations = boundary_values(species, background_concentrations)
    if mf6_executable is None:
        mf6_executable = executable_path()
    if len(logical_steps_per_period) != 2 or any(steps <= 0 for steps in logical_steps_per_period):
        raise ValueError("logical_steps_per_period must contain two positive values")
    delr = length / ncol
    delc = width / nrow
    pulse = np.asarray(pulse_concentrations, dtype=float).ravel()
    background = np.asarray(background_concentrations, dtype=float).ravel()
    multiplier = 2 if strang_half_steps else 1
    simulation = flopy.mf6.MFSimulation(
        sim_name="splitting_redox_2d",
        sim_ws=str(workspace),
        exe_name=mf6_executable,
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        simulation,
        time_units="DAYS",
        nper=2,
        perioddata=[
            (pulse_duration, logical_steps_per_period[0] * multiplier, 1.0),
            (flush_duration, logical_steps_per_period[1] * multiplier, 1.0),
        ],
    )
    gwf = flopy.mf6.ModflowGwf(simulation, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        complexity="MODERATE",
        outer_maximum=100,
        outer_dvclose=1e-09,
        inner_maximum=200,
        inner_dvclose=1e-10,
        rcloserecord=1e-09,
        linear_acceleration="BICGSTAB",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    discretization = dict(
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=top,
        botm=botm,
    )
    flopy.mf6.ModflowGwfdis(gwf, **discretization)
    initial_head = np.tile(np.linspace(inlet_head, outlet_head, ncol), (nrow, 1))
    flopy.mf6.ModflowGwfic(gwf, strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf, icelltype=0, k=hydraulic_conductivity, save_specific_discharge=True
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0)
    source_rows = range(nrow // 3, 2 * nrow // 3)
    pulse_records = []
    flush_records = []
    for row in range(nrow):
        row_concentration = pulse if row in source_rows else background
        pulse_records.append(((0, row, 0), inlet_head, *row_concentration))
        flush_records.append(((0, row, 0), inlet_head, *background))
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="INLET",
        auxiliary=species,
        stress_period_data={0: pulse_records, 1: flush_records},
    )
    outlet_records = [((0, row, ncol - 1), outlet_head) for row in range(nrow)]
    flopy.mf6.ModflowGwfchd(gwf, pname="OUTLET", stress_period_data={0: outlet_records})
    source_records = [((0, row, column), 0.0) for row in range(nrow) for column in range(ncol)]
    for species_name, concentration in initial_fields.items():
        gwt_name = get_gwt_model_name(species_name)
        gwt = flopy.mf6.ModflowGwt(
            simulation, modelname=gwt_name, save_flows=True, model_nam_file=f"{gwt_name}.nam"
        )
        transport_ims = flopy.mf6.ModflowIms(
            simulation,
            pname=f"{gwt_name}_ims",
            filename=f"{gwt_name}.ims",
            outer_maximum=100,
            outer_dvclose=1e-09,
            inner_maximum=200,
            inner_dvclose=1e-10,
            rcloserecord=1e-09,
            linear_acceleration="BICGSTAB",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(gwt, **discretization)
        flopy.mf6.ModflowGwtic(gwt, strt=concentration.reshape(1, nrow, ncol))
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(gwt, xt3d_off=True, alh=alh, ath1=ath1, diffc=diffc)
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        flopy.mf6.ModflowGwtsrc(
            gwt, pname="SRC", maxbound=nxyz, stress_period_data={0: source_records}
        )
        flopy.mf6.ModflowGwtssm(gwt, sources=[("INLET", "AUX", species_name)])
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )
    simulation.write_simulation(silent=True)
    return simulation
