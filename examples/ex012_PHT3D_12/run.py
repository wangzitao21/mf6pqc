"""Run the ex012_PHT3D_12 reactive-transport benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex012_PHT3D_12.modflow_model import build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    FeedbackOptions,
    OutputOptions,
    SimulationConfig,
)

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
NLAY = 1
NROW = 1
NCOL = 212
NXYZ = NLAY * NROW * NCOL
DELR = 0.001
DELC = 1.0
TOP = 0.0
BOTM = -0.00038
POROSITY = 0.42
INFLOW_RATE = 0.00048
ALH = 7.5e-05
TRANSPORT_SUBSTEPS = 16
FLOW_PERIOD_DATA = ((0.08333, 260), (1.03919, 3132))
PULSE_END = FLOW_PERIOD_DATA[0][0]
PHT3D_MOBILE_COMPONENTS = frozenset({"H2O", "H", "O", "Tracer", "U", "Na", "N", "F"})
INITIAL_HEAD = 0.0
HYDRAULIC_CONDUCTIVITY = 1.0
DIFFC = 0.0
OUTLET_HEAD = 0.0


def cell_centers_x() -> np.ndarray:
    return (np.arange(NCOL, dtype=float) + 0.5) * DELR


def flow_step_end_times() -> np.ndarray:
    step_ends: list[np.ndarray] = []
    period_start = 0.0
    for period_length, flow_step_count in FLOW_PERIOD_DATA:
        step_ends.append(
            period_start
            + np.arange(1, flow_step_count + 1, dtype=float) * (period_length / flow_step_count)
        )
        period_start += period_length
    return np.unique(np.concatenate(step_ends).astype(np.float32)).astype(float)


def coupling_step_end_times(output_times: np.ndarray) -> np.ndarray:
    events = np.concatenate([flow_step_end_times(), np.asarray(output_times, dtype=float)]).astype(
        np.float32
    )
    events = np.unique(events).astype(float)
    return events[events > 0.0]


def coupling_period_data(output_times: np.ndarray) -> list[tuple[float, int, float]]:
    step_ends = coupling_step_end_times(output_times)
    step_lengths = np.diff(np.concatenate(([0.0], step_ends)))
    if np.any(step_lengths <= 0.0):
        raise ValueError("PHT3D coupling times must be strictly increasing")
    return [(float(length), TRANSPORT_SUBSTEPS, 1.0) for length in step_lengths]


def save_step_numbers(target_days: tuple[float, ...], coupling_times: np.ndarray) -> list[int]:
    """Map requested days to the corresponding transport-step numbers."""
    saved_steps: list[int] = []
    for target in target_days:
        index = int(np.searchsorted(coupling_times, target))
        if index == len(coupling_times) or not np.isclose(
            coupling_times[index], target, rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"Official output time {target} is not a coupling step")
        saved_steps.append(index + 1)
    return [step * TRANSPORT_SUBSTEPS for step in saved_steps]


def main() -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    save_hours = (1.5, 3.0, 12.5)
    reference = np.load(INPUT_DIR / "PHT3D_12_results.npy", allow_pickle=False)
    save_target_days = tuple(reference["actual_hours"] / 24.0)
    coupling_times = coupling_step_end_times(reference["output_days"])
    period_data = coupling_period_data(reference["output_days"])
    if len(period_data) != 4388:
        raise ValueError(
            f"Official E12 schedule must contain 4,388 transport intervals; reconstructed {len(period_data)}"
        )
    save_steps = save_step_numbers(save_target_days, coupling_times)
    flow_reaction_steps = save_step_numbers(tuple(flow_step_end_times()), coupling_times)
    reaction_steps = sorted(set(flow_reaction_steps) | set(save_steps))
    simulation_config = SimulationConfig(
        case_name="ex012",
        nxyz=NXYZ,
        nthreads=6,
        paths=BackendPaths(
            database=INPUT_DIR / "database.dat",
            chemistry_input=INPUT_DIR / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=25.0,
            pressure_atm=1.0,
            porosity=POROSITY,
            saturation=1.0,
            density_kg_per_litre=1.0,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=True,
            use_solution_density_volume=False,
            signed_components=(),
        ),
        feedback=FeedbackOptions(update_porosity_and_k=False, update_density=False),
        output=OutputOptions(save_steps=save_steps, progress_interval=100 * TRANSPORT_SUBSTEPS),
        reaction_steps=reaction_steps,
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map={"solution": 0, "surface": 1})
        species = simulator.get_components()
        pulse_concentrations = simulator.get_initial_concentrations(1)
        chase_concentrations = simulator.get_initial_concentrations(0)
        charge_index = species.index("Charge")
        initial_concentrations[charge_index * NXYZ : (charge_index + 1) * NXYZ] = 0.0
        pulse_concentrations[charge_index] = 0.0
        chase_concentrations[charge_index] = 0.0
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            pulse_concentrations=pulse_concentrations,
            chase_concentrations=chase_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            period_data=period_data,
            porosity=POROSITY,
            hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
            initial_head=INITIAL_HEAD,
            outlet_head=OUTLET_HEAD,
            inflow_rate=INFLOW_RATE,
            alh=ALH,
            diffc=DIFFC,
            pht3d_mobile_components=PHT3D_MOBILE_COMPONENTS,
            pulse_end=PULSE_END,
        )
        simulator.run()
        simulator.save_results()
        print(
            f"Saved target hours {save_hours} at exact PHT3D coupling steps {[step // TRANSPORT_SUBSTEPS for step in save_steps]}; {len(flow_reaction_steps)} official flow-step reactions plus {len(set(save_steps) - set(flow_reaction_steps))} save points over {len(period_data) * TRANSPORT_SUBSTEPS} MF6 transport steps."
        )


if __name__ == "__main__":
    configure_logging()
    main()
