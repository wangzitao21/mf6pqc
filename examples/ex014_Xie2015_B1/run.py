"""Run the ex014_Xie2015_B1 reactive-transport benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex014_Xie2015_B1.modflow_model import build_model
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

CASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = CASE_DIR / "input_data"
NLAY = 1
NROW = 1
NCOL = 81
NXYZ = NLAY * NROW * NCOL
POROSITY = 0.35
DELR = [0.0125] + [0.025] * (NCOL - 2) + [0.0125]
DELC = [1.0]
TOP = 1.0
BOTM = 0
SNIA_PERIOD_DATA = [(365.0 * 10, 10000, 1.0), (365.0 * 110, 110000, 1.0), (365.0 * 380, 38000, 1.0)]
ALH = 0.0
ATH1 = ALH / 10
DIFFC = 0.0
BOUNDARY_DISTANCE = DELR[0] / 2
INLET_HEAD = 0.007
OUTLET_HEAD = 0.0
VERTICAL_CONDUCTIVITY_RATIO = 0.1
HYDRAULIC_CONDUCTIVITY = 1.16e-4 * 86400.0
INITIAL_HEAD = 0.0
CALCITE_MOLAR_VOLUME = 100.0894 / 2710.0
INITIAL_CALCITE_VOLUME_FRACTION = 0.30
DEFAULT_MAX_STEP_YEARS = 0.3


class B1Backend(NativeBackendFactory):
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


def time_discretization(years=500.0, max_step_years=DEFAULT_MAX_STEP_YEARS, *, method="implicit"):
    if (
        not np.isfinite(years)
        or years <= 0
        or not np.isfinite(max_step_years)
        or max_step_years <= 0
    ):
        raise ValueError("Simulation duration and maximum time step must be positive and finite")
    ends = sorted({y for y in (1, 10, 100, 120, 130, 150, 200, 300, 500, years) if y <= years})
    periods, time_steps = [], []
    previous = 0.0
    for end in ends:
        step = max_step_years if method == "implicit" else (0.001 if previous < 120 else 0.01)
        count, multiplier = int(np.ceil((end - previous) / step)), 1.0
        if method == "implicit" and previous == 0 and end == 1:
            count, multiplier = 60, 1.15
            while (1 - 1 / multiplier) / (1 - multiplier ** (-count)) > step:
                multiplier = np.sqrt(multiplier)
                count *= 2
        duration = (end - previous) * 365.0
        weights = multiplier ** (np.arange(count, dtype=float) - count + 1)
        durations = duration * weights / weights.sum()
        durations[-1] += duration - durations.sum()
        periods.append((duration, count, multiplier))
        time_steps.extend(durations)
        previous = end
    times = np.cumsum(time_steps)
    targets = sorted({*ends, *np.arange(1, min(years, 150) + 1), *np.arange(160, years + 1, 10)})
    save_steps = sorted({int(np.abs(times - y * 365).argmin()) + 1 for y in targets})
    return periods, save_steps


def main(argv=None) -> None:
    """Configure, build, run, and save this reactive-transport benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("implicit", "snia"), default="implicit")
    parser.add_argument("--years", type=float, default=500.0)
    parser.add_argument(
        "--max-step-years",
        type=float,
        default=DEFAULT_MAX_STEP_YEARS,
        help="Maximum implicit step; use 0.15 or 0.075 for refinement checks",
    )
    parser.add_argument("--threads", type=int, default=4, help="Native PhreeqcRM threads")
    args = parser.parse_args(argv)
    periods, save_steps = time_discretization(args.years, args.max_step_years, method=args.method)
    workspace = runtime_path(__file__, "simulation")
    output_dir = runtime_path(__file__, "output")
    ic_mapping = {"solution": 0, "kinetics": 1}
    simulation_config = SimulationConfig(
        case_name="ex014",
        nxyz=NXYZ,
        nthreads=args.threads,
        backend_factory=B1Backend(),
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
            transport_water_component=False,
            use_solution_density_volume=False,
        ),
        feedback=FeedbackOptions(
            update_porosity_and_k=True,
            update_density=False,
            mineral_molar_volumes={"Calcite": CALCITE_MOLAR_VOLUME},
            vertical_to_horizontal_k_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            fail_on_porosity_clipping=True,
            boundary_conductance_updates={
                "BUSHUI": {"cell_index": 0, "distance": BOUNDARY_DISTANCE},
                "GHB_RIGHT": {"cell_index": -1, "distance": BOUNDARY_DISTANCE},
            },
        ),
        output=OutputOptions(save_steps=save_steps),
        fail_on_modflow_nonconvergence=True,
    )
    parameters = re.findall(
        r"(?im)^\s*-parms\s+([\d.eE+\-]+)\s*$", (INPUT_DIR / "input.pqi").read_text()
    )
    if len(parameters) != 1:
        raise ValueError("B1 input must define one Calcite rate constant")
    simulation_config.implicit = ImplicitOptions(
        reactions=(
            KineticReaction(
                "Calcite",
                {"Ca": 1, "C": 1, "O": 3},
                float(parameters[0]) * 86400.0,
                surface_exponent=2 / 3,
            ),
        )
    )
    hydraulic_conductivity = np.full((NLAY, NROW, NCOL), HYDRAULIC_CONDUCTIVITY)
    with MF6PQC.from_config(simulation_config) as simulator:
        initial_concentrations = simulator.setup(ic_map=ic_mapping)
        species = simulator.get_components()
        inflow_concentrations = simulator.get_initial_concentrations(1)
        build_model(
            workspace=workspace,
            species=species,
            initial_concentrations=initial_concentrations,
            inflow_concentrations=inflow_concentrations,
            nlay=NLAY,
            nrow=NROW,
            ncol=NCOL,
            delr=DELR,
            delc=DELC,
            top=TOP,
            botm=BOTM,
            period_data=periods,
            porosity=POROSITY,
            hydraulic_conductivity=hydraulic_conductivity,
            vertical_conductivity_ratio=VERTICAL_CONDUCTIVITY_RATIO,
            initial_head=INITIAL_HEAD,
            inlet_head=INLET_HEAD,
            outlet_head=OUTLET_HEAD,
            alh=ALH,
            ath1=ATH1,
            diffc=DIFFC,
            boundary_distance=BOUNDARY_DISTANCE,
        )
        simulator.run("Implicit" if args.method == "implicit" else "SNIA")
        simulator.save_results()
        if args.method == "implicit":
            manifest_path = output_dir / "results_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["b1_implicit"] = {
                "max_step_years": args.max_step_years,
                "native_rm_threads": simulator.phreeqc_rm.GetThreadCount(),
                "checks": simulator.implicit_diagnostics,
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print("ex014 done.")


if __name__ == "__main__":
    configure_logging()
    main()
