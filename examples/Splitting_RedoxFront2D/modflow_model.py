from __future__ import annotations

from collections.abc import Sequence

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


MATRIX_OXIDANT_CAPACITY = 2.0e-4
LENS_OXIDANT_CAPACITY = 8.0e-4


def hydraulic_conductivity_field(nrow: int, ncol: int) -> np.ndarray:
    """Return a deterministic channel-and-matrix conductivity field (m/day)."""
    rows, columns = np.indices((nrow, ncol))
    channel_center = 0.50 * (nrow - 1) + 1.8 * np.sin(
        2.0 * np.pi * columns / max(ncol - 1, 1)
    )
    distance = np.abs(rows - channel_center)
    field = np.full((nrow, ncol), 0.45, dtype=float)
    field[distance <= 1.25] = 1.35
    field[(columns > ncol // 2) & (distance > 3.5)] = 0.25
    return field


def reactive_lens_mask(nrow: int, ncol: int) -> np.ndarray:
    """Return the diagonal high-activity biogeochemical lens."""
    rows, columns = np.indices((nrow, ncol))
    lens_center = 0.72 * (nrow - 1) - 0.34 * columns
    return (
        (columns >= int(0.28 * ncol))
        & (columns <= int(0.76 * ncol))
        & (np.abs(rows - lens_center) <= 1.35)
    )


def oxidant_capacity_field(nrow: int, ncol: int) -> np.ndarray:
    """Return ferric-oxide electron-equivalent capacity per chemistry cell."""
    return np.where(
        reactive_lens_mask(nrow, ncol),
        LENS_OXIDANT_CAPACITY,
        MATRIX_OXIDANT_CAPACITY,
    )


def _component_fields(
    species_list: Sequence[str], initial_conc: np.ndarray, nxyz: int
) -> dict[str, np.ndarray]:
    values = np.asarray(initial_conc, dtype=float).ravel()
    expected = len(species_list) * nxyz
    if values.size != expected:
        raise ValueError(
            f"initial_conc has {values.size} entries; expected {expected}"
        )
    return {
        component: values[index * nxyz : (index + 1) * nxyz]
        for index, component in enumerate(species_list)
    }


def build_transport_model(
    sim_ws: str,
    species_list: list[str],
    initial_conc: np.ndarray,
    pulse_conc: np.ndarray,
    background_conc: np.ndarray,
    *,
    nrow: int = 12,
    ncol: int = 24,
    length: float = 8.0,
    width: float = 4.0,
    porosity: float = 0.35,
    pulse_duration: float = 8.0,
    flush_duration: float = 4.0,
    logical_steps_per_period: tuple[int, int] = (1, 1),
    strang_half_steps: bool = False,
) -> dict[str, np.ndarray]:
    """Build the two-dimensional pulse/flush flow and transport system."""
    if len(logical_steps_per_period) != 2 or any(
        steps <= 0 for steps in logical_steps_per_period
    ):
        raise ValueError("logical_steps_per_period must contain two positive values")
    nxyz = nrow * ncol
    delr = length / ncol
    delc = width / nrow
    conductivity = hydraulic_conductivity_field(nrow, ncol)
    component_fields = _component_fields(species_list, initial_conc, nxyz)
    pulse = np.asarray(pulse_conc, dtype=float).ravel()
    background = np.asarray(background_conc, dtype=float).ravel()
    if pulse.size != len(species_list) or background.size != len(species_list):
        raise ValueError("boundary concentration vectors must match species_list")

    multiplier = 2 if strang_half_steps else 1
    simulation = flopy.mf6.MFSimulation(
        sim_name="splitting_redox_2d",
        sim_ws=sim_ws,
        exe_name="./bin/mf6.7.0/mf6.exe",
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

    gwf = flopy.mf6.ModflowGwf(
        simulation, modelname="gwf_model", save_flows=True
    )
    flow_ims = flopy.mf6.ModflowIms(
        simulation,
        pname="flow_ims",
        complexity="MODERATE",
        outer_maximum=100,
        outer_dvclose=1.0e-9,
        inner_maximum=200,
        inner_dvclose=1.0e-10,
        rcloserecord=1.0e-9,
        linear_acceleration="BICGSTAB",
    )
    simulation.register_ims_package(flow_ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=1,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=1.0,
        botm=0.0,
    )
    initial_head = np.tile(np.linspace(0.8, 0.0, ncol), (nrow, 1))
    flopy.mf6.ModflowGwfic(gwf, strt=initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        icelltype=0,
        k=conductivity,
        save_specific_discharge=True,
    )
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.0, sy=0.0)

    source_rows = range(nrow // 3, 2 * nrow // 3)
    pulse_records = []
    flush_records = []
    for row in range(nrow):
        row_concentration = pulse if row in source_rows else background
        pulse_records.append(((0, row, 0), 0.8, *row_concentration))
        flush_records.append(((0, row, 0), 0.8, *background))
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="INLET",
        auxiliary=species_list,
        stress_period_data={0: pulse_records, 1: flush_records},
    )
    outlet_records = [((0, row, ncol - 1), 0.0) for row in range(nrow)]
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="OUTLET",
        stress_period_data={0: outlet_records},
    )

    source_records = [
        ((0, row, column), 0.0)
        for row in range(nrow)
        for column in range(ncol)
    ]
    for component, starting_concentration in component_fields.items():
        gwt_name = get_gwt_model_name(component)
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
            outer_dvclose=1.0e-9,
            inner_maximum=200,
            inner_dvclose=1.0e-10,
            rcloserecord=1.0e-9,
            linear_acceleration="BICGSTAB",
        )
        simulation.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=1,
            nrow=nrow,
            ncol=ncol,
            delr=delr,
            delc=delc,
            top=1.0,
            botm=0.0,
        )
        flopy.mf6.ModflowGwtic(
            gwt, strt=starting_concentration.reshape(1, nrow, ncol)
        )
        flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=0.12,
            ath1=0.012,
            diffc=0.0,
        )
        flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)
        flopy.mf6.ModflowGwtsrc(
            gwt,
            pname="SRC",
            maxbound=nxyz,
            stress_period_data={0: source_records},
        )
        flopy.mf6.ModflowGwtssm(
            gwt, sources=[("INLET", "AUX", component)]
        )
        flopy.mf6.ModflowGwfgwt(
            simulation,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )

    simulation.write_simulation(silent=True)
    return {
        "hydraulic_conductivity": conductivity,
        "reactive_lens": reactive_lens_mask(nrow, ncol),
    }
