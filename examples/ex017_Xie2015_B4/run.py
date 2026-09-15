"""Run the ex017_Xie2015_B4 reactive-transport benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex017_Xie2015_B4.modflow_model import build_model
from example_utils import configure_logging, library_path, runtime_path

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    FeedbackOptions,
    ImplicitOptions,
    KineticReaction,
    OutputOptions,
    SimulationConfig,
)
from mf6pqc.backends import NativeBackendFactory


class QuietNativeBackend(NativeBackendFactory):
    def create_phreeqcrm(self, nxyz, nthreads):
        native = super().create_phreeqcrm(nxyz, nthreads)

        class QuietRM:
            def __getattr__(self, name):
                return getattr(native, name)

            def OpenFiles(self):
                return 0

            def CloseFiles(self):
                return 0

        return QuietRM()


def schedule(years, maximum, targets):
    if not np.isfinite(years) or not np.isfinite(maximum) or min(years, maximum) <= 0:
        raise ValueError("Duration and maximum time step must be positive and finite")
    ends = sorted({v for v in [1.0, *targets, years] if 0 < v <= years})
    periods, steps, save_steps = [], [], []
    previous = 0.0

    def append_period(duration, count, multiplier):
        weights = multiplier ** (np.arange(count, dtype=float) - count + 1)
        durations = duration * 365 * weights / weights.sum()
        durations[-1] += duration * 365 - durations.sum()
        periods.append((duration * 365, count, multiplier))
        steps.extend(durations)

    for end in ends:
        duration = end - previous
        if previous == 0:
            count = max(int(np.ceil(duration / maximum)), 100)
            multiplier = 1.12
            while (1 - 1 / multiplier) / (1 - multiplier ** (-count)) * duration > maximum:
                multiplier = np.sqrt(multiplier)
                count *= 2
            append_period(duration, count, multiplier)
        else:
            next_step = steps[-1] / 365 * 1.12
            ramp, count = 0.0, 0
            while next_step < maximum and ramp + next_step < duration:
                ramp += next_step
                count += 1
                next_step *= 1.12
            if count:
                append_period(ramp, count, 1.12)
            remaining = duration - ramp
            append_period(remaining, max(1, int(np.ceil(remaining / maximum))), 1.0)
        save_steps.append(len(steps))
        previous = end
    times = np.cumsum(steps)

    stride = max(1, int(np.ceil(len(steps) / 300)))
    save_steps = sorted({*save_steps, *range(stride, len(steps) + 1, stride)})
    return periods, save_steps, times


def normalize_min3p_totals(values, components, solution):
    values = np.asarray(values, dtype=float)
    ca = values[components.index("Ca")]
    if np.any(ca <= 0):
        raise ValueError("The MIN3P concentration conversion requires positive Ca")
    result = values * (solution["ca+2"]["value"] / ca)
    for source, component in (
        ("ca+2", "Ca"),
        ("co3-2", "C"),
        ("so4-2", "S"),
        ("na+1", "Na"),
        ("al+3", "Al"),
        ("k+1", "K"),
    ):
        if source in solution and solution[source]["constraint"] != "charge":
            expected = solution[source]["value"]
            if not np.allclose(
                result[components.index(component)], expected, rtol=1e-8, atol=1e-12
            ):
                raise ValueError(f"Initial {component} differs from the MIN3P total")
    if "fe+2" in solution:
        expected = solution["fe+2"]["value"] + solution["fe+3"]["value"]
        if not np.allclose(result[components.index("Fe")], expected, rtol=1e-8, atol=1e-12):
            raise ValueError("Initial total Fe differs from the MIN3P input")
    return result


def main(argv=None):
    case = Path(__file__).resolve().parent
    input_dir = case / "input_data"
    parameters = json.loads((input_dir / "min3p_parameters.json").read_text())
    parser = argparse.ArgumentParser(
        description="Xie B4: implicit kinetic coupling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--method", choices=("implicit", "snia"), default="implicit")
    parser.add_argument("--years", type=float, default=parameters["final_years"])
    parser.add_argument(
        "--max-step-years",
        type=float,
        default=1.0,
        help="Maximum physical time step in years; verify accuracy by time refinement",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--kinetic-atol", type=float, default=1e-6)
    args = parser.parse_args(argv)
    maximum = args.max_step_years if args.method == "implicit" else min(args.max_step_years, 0.001)
    periods, saves, _ = schedule(args.years, maximum, parameters["reference_times_years"])
    n = 81
    phi = parameters["porosity"]
    conductivity = parameters["hydraulic_conductivity_m_per_s"] * 86400
    d0 = parameters["diffusion_m2_per_s"] * 86400
    widths = np.r_[0.0125, np.full(n - 2, 0.025), 0.0125]
    reactive = np.ones(n)
    reactive[[0, -1]] = 0
    entries = parameters["minerals"]
    reactions = tuple(
        KineticReaction(
            e["name"],
            e["stoichiometry"],
            reactive * e["rate_constant_mol_bulk_per_second"] * 86400,
            surface_exponent=e["surface_exponent"],
            minimum_amount=reactive * e["minimum_amount_mol_bulk"],
            saturation_index_heading="SI_" + e["name"],
            reference_amount=e["initial_amount_mol_bulk"] if e["surface_exponent"] else None,
        )
        for e in entries
    )
    workspace, output = runtime_path(__file__, "simulation"), runtime_path(__file__, "output")
    feedback = FeedbackOptions(
        update_porosity_and_k=True,
        update_diffusion=True,
        porosity_update_mask=reactive.astype(bool),
        mineral_molar_volumes={e["name"]: e["molar_volume_l_per_mol"] for e in entries},
        vertical_to_horizontal_k_ratio=1.0,
        fail_on_porosity_clipping=True,
        boundary_conductance_updates={
            "BUSHUI": {"cell_index": 0, "distance": widths[0] / 2},
            "GHB_RIGHT": {"cell_index": -1, "distance": widths[-1] / 2},
        },
    )
    config = SimulationConfig(
        case_name=case.name,
        nxyz=n,
        nthreads=args.threads,
        backend_factory=QuietNativeBackend(),
        paths=BackendPaths(
            database=input_dir / "database.dat",
            chemistry_input=input_dir / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output,
        ),
        fields=CellFields(
            temperature_c=25.0,
            pressure_atm=1.0,
            porosity=phi,
            saturation=1.0,
            density_kg_per_litre=1.0,
            free_water_diffusion_model_units=d0,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=False,
            use_solution_density_volume=False,
        ),
        feedback=feedback,
        output=OutputOptions(save_steps=saves, progress_interval=1000),
        implicit=ImplicitOptions(
            reactions=reactions,
            dense_limit=1024,
            absolute_tolerance=args.kinetic_atol,
            derivative_refresh=1,
            chemical_jacobian="species",
            maximum_iterations=80,
            porosity_coupling="lagged",
            predict_porosity=False,
        ),
        fail_on_modflow_nonconvergence=True,
    )
    with MF6PQC.from_config(config) as sim:
        initial = sim.setup(ic_map={"solution": 0, "kinetics": 1})

        initial = normalize_min3p_totals(
            initial.reshape(sim.ncomps, n), sim.components, parameters["solutions"]["0"]
        )
        inlet = normalize_min3p_totals(
            sim.get_initial_concentrations(1), sim.components, parameters["solutions"]["1"]
        )
        sim.phreeqc_rm.SetConcentrations(initial.ravel())
        sim.phreeqc_rm.SetTimeStep(0)
        sim.phreeqc_rm.RunCells()
        from mf6pqc.coupling.common import update_selected_output

        update_selected_output(sim)
        sim.results[0] = sim.selected_output
        initial = initial.ravel()

        lines = ["KINETICS_MODIFY 0"] + [f"-component {e['name']}\n-m 0" for e in entries]
        sim.phreeqc_rm.RunString(True, False, False, "\n".join(lines) + "\nEND\n")
        sim.phreeqc_rm.SetTimeStep(0)
        sim.phreeqc_rm.RunCells()
        from mf6pqc.coupling.common import update_selected_output

        update_selected_output(sim)
        sim.results[0] = sim.selected_output
        build_model(
            workspace=workspace,
            species=sim.get_components(),
            initial_concentrations=initial,
            inflow_concentrations=inlet,
            nlay=1,
            nrow=1,
            ncol=n,
            delr=widths,
            delc=[1.0],
            top=1.0,
            botm=0.0,
            period_data=periods,
            porosity=phi,
            hydraulic_conductivity=np.full((1, 1, n), conductivity),
            vertical_conductivity_ratio=1.0,
            initial_head=0.0,
            inlet_head=0.0,
            outlet_head=0.0,
            alh=0.0,
            ath1=0.0,
            diffc=np.cbrt(phi) * d0,
            boundary_distance=widths[0] / 2,
            diffusion_boundary=True,
            node_coordinates=np.linspace(0, 2, n),
        )
        sim.run("Implicit" if args.method == "implicit" else "SNIA")
        sim.save_results()
        manifest_path = output / "results_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["min3p_benchmark"] = dict(
            parameters, max_step_years=maximum, method=args.method, native_threads=args.threads
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"B4 complete: {output}")


if __name__ == "__main__":
    configure_logging()
    main()
