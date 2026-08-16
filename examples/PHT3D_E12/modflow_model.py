"""MODFLOW 6 model for PHT3D Example 12."""

from __future__ import annotations

from pathlib import Path

import flopy
import numpy as np

from mf6pqc.utils import get_gwt_model_name


NLAY = 1
NROW = 1
NCOL = 212
NXYZ = NLAY * NROW * NCOL
DELR = 1.0e-3
DELC = 1.0
TOP = 0.0
BOTM = -3.8e-4
POROSITY = 0.42
FLOW_RATE = 4.8e-4
LONGITUDINAL_DISPERSIVITY = 7.5e-5
TRANSPORT_SUBSTEPS = 16
FLOW_PERIOD_DATA = ((0.08333, 260), (1.03919, 3132))
PULSE_END = FLOW_PERIOD_DATA[0][0]


def cell_centers_x() -> np.ndarray:
    return (np.arange(NCOL, dtype=float) + 0.5) * DELR


def coupling_step_end_times(output_times: np.ndarray) -> np.ndarray:
    """Return the outer transport/reaction step ends used by PHT3D.

    PHT3D calls PHREEQC at the end of every MODFLOW flow step and whenever a
    requested ``TIMPRS`` output falls inside a flow step.  ``PERCEL`` controls
    the ULTIMATE advection calculation; it does *not* create extra outer
    PHREEQC calls.  This distinction is visible in the official
    ``PHT3D001.MAS`` file, which contains 4,388 completed steps.

    MT3D stores elapsed time in single precision.  Converting both sets of
    events to float32 before taking their union is therefore important: five
    requested outputs coincide with flow-step ends only after the same
    rounding used by PHT3D.
    """
    flow_step_ends: list[np.ndarray] = []
    period_start = 0.0
    for period_length, flow_step_count in FLOW_PERIOD_DATA:
        flow_step_ends.append(
            period_start
            + np.arange(1, flow_step_count + 1, dtype=float)
            * (period_length / flow_step_count)
        )
        period_start += period_length

    events = np.concatenate(
        [*flow_step_ends, np.asarray(output_times, dtype=float)]
    ).astype(np.float32)
    events = np.unique(events).astype(float)
    return events[events > 0.0]


def coupling_period_data(output_times: np.ndarray) -> list[tuple[float, int, float]]:
    """Represent each PHT3D coupling interval with refined MF6 transport.

    MODFLOW 6 TVD is more time-step diffusive than MT3D's ULTIMATE scheme for
    this near-unit-Courant column.  A conservative-tracer convergence study
    gives NRMSE values of 6.3%, 1.9%, 1.2%, and 0.9% for 1, 8, 16, and 32
    transport substeps, respectively.  Sixteen is the practical knee of that
    curve.  PHREEQC still runs only once at the end of each outer interval.
    """
    step_ends = coupling_step_end_times(output_times)
    step_lengths = np.diff(np.concatenate(([0.0], step_ends)))
    if np.any(step_lengths <= 0.0):
        raise ValueError("PHT3D coupling times must be strictly increasing")
    return [
        (float(length), TRANSPORT_SUBSTEPS, 1.0) for length in step_lengths
    ]


def _component_fields(
    species_list: list[str], initial_concentrations: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: initial_concentrations[index * NXYZ : (index + 1) * NXYZ]
        for index, name in enumerate(species_list)
    }


def transport_model(
    *,
    sim_ws: str | Path,
    species_list: list[str],
    initial_conc: np.ndarray,
    pulse_concentrations: np.ndarray,
    chase_concentrations: np.ndarray,
    period_data: list[tuple[float, int, float]],
    mf6_exe: str | Path,
    longitudinal_dispersivity: float = LONGITUDINAL_DISPERSIVITY,
) -> None:
    """Build the official column using PHT3D's reaction-event schedule."""
    sim = flopy.mf6.MFSimulation(
        sim_name="model",
        sim_ws=str(sim_ws),
        exe_name=str(mf6_exe),
        verbosity_level=0,
    )
    flopy.mf6.ModflowTdis(
        sim,
        time_units="DAYS",
        nper=len(period_data),
        perioddata=period_data,
    )

    gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf_model", save_flows=True)
    flow_ims = flopy.mf6.ModflowIms(
        sim,
        pname="flow_ims",
        print_option="NONE",
        complexity="SIMPLE",
        outer_dvclose=1.0e-10,
        inner_dvclose=1.0e-11,
        rcloserecord=1.0e-10,
        linear_acceleration="CG",
        filename="flow.ims",
    )
    sim.register_ims_package(flow_ims, [gwf.name])
    flopy.mf6.ModflowGwfdis(
        gwf,
        nlay=NLAY,
        nrow=NROW,
        ncol=NCOL,
        delr=DELR,
        delc=DELC,
        top=TOP,
        botm=BOTM,
    )
    flopy.mf6.ModflowGwfic(gwf, strt=0.0)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        save_flows=True,
        save_specific_discharge=True,
        icelltype=0,
        k=1.0,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        iconvert=0,
        ss=0.0,
        sy=0.0,
        steady_state={0: True},
    )

    pulse = [[(0, 0, 0), FLOW_RATE, *pulse_concentrations]]
    chase = [[(0, 0, 0), FLOW_RATE, *chase_concentrations]]
    pulse_end_period = next(
        index
        for index, elapsed in enumerate(
            np.cumsum([entry[0] for entry in period_data])
        )
        if np.isclose(elapsed, PULSE_END)
    ) + 1
    flopy.mf6.ModflowGwfwel(
        gwf,
        pname="WEL-INLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={0: pulse, pulse_end_period: chase},
        auxiliary=species_list,
    )
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="CHD-OUTLET",
        save_flows=True,
        maxbound=1,
        stress_period_data={0: [[(0, 0, NCOL - 1), 0.0]]},
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        budget_filerecord="gwf_model.bud",
        head_filerecord="gwf_model.hds",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    for species, concentration in _component_fields(
        species_list, initial_conc
    ).items():
        gwt_name = get_gwt_model_name(species)
        # H and O include the water inventory (roughly 111 and 55 mol/L).
        # With very short TIMPRS-generated steps, roundoff in those two
        # equations cannot satisfy a 1e-12 absolute mass-rate residual even
        # after the concentration update is exactly zero.  A 1e-9 residual
        # is still more than seven orders below their inlet mass rates.  Keep
        # the stricter criterion for low-concentration Tracer and U.
        transport_rclose = (
            1.0e-9 if species in {"H2O", "H", "O"} else 1.0e-12
        )
        gwt = flopy.mf6.ModflowGwt(
            sim,
            modelname=gwt_name,
            save_flows=False,
            model_nam_file=f"{gwt_name}.nam",
        )
        transport_ims = flopy.mf6.ModflowIms(
            sim,
            print_option="NONE",
            complexity="SIMPLE",
            # Tracer enters at only 1e-6 mol/L.  The usual 1e-8/1e-9
            # concentration tolerances truncate its advancing front and cause
            # artificial mass loss, while the 5e-5 mol/L U front can hide the
            # same problem.  Use absolute tolerances below the transported
            # concentration scale.
            outer_dvclose=1.0e-11,
            outer_maximum=200,
            inner_maximum=1000,
            inner_dvclose=1.0e-12,
            # The tracer mass-source rate is only about 4.8e-10 mol/day.
            # An rclose of 1e-8 therefore permits a residual larger than the
            # entire source and accumulated an 8% mass deficit by 1.5 h.
            rcloserecord=transport_rclose,
            linear_acceleration="BICGSTAB",
            filename=f"{gwt_name}.ims",
        )
        sim.register_ims_package(transport_ims, [gwt.name])
        flopy.mf6.ModflowGwtdis(
            gwt,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            filename=f"{gwt_name}.dis",
        )
        flopy.mf6.ModflowGwtic(
            gwt, strt=concentration, filename=f"{gwt_name}.ic"
        )
        flopy.mf6.ModflowGwtadv(
            gwt, scheme="TVD", filename=f"{gwt_name}.adv"
        )
        flopy.mf6.ModflowGwtdsp(
            gwt,
            xt3d_off=True,
            alh=longitudinal_dispersivity,
            ath1=0.1 * longitudinal_dispersivity,
            diffc=0.0,
            filename=f"{gwt_name}.dsp",
        )
        flopy.mf6.ModflowGwtmst(
            gwt, porosity=POROSITY, filename=f"{gwt_name}.mst"
        )
        flopy.mf6.ModflowGwtssm(
            gwt,
            sources=[("WEL-INLET", "AUX", species)],
            filename=f"{gwt_name}.ssm",
        )
        flopy.mf6.ModflowGwtoc(
            gwt,
            budget_filerecord=f"{gwt_name}.cbc",
            concentration_filerecord=f"{gwt_name}.ucn",
            saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        )
        flopy.mf6.ModflowGwfgwt(
            sim,
            exgtype="GWF6-GWT6",
            exgmnamea=gwf.name,
            exgmnameb=gwt.name,
            filename=f"{gwt_name}.gwfgwt",
        )

    sim.write_simulation(silent=True)
