"""Configure, run, and validate the brine feedback scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.dont_write_bytecode = True
EXAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import numpy as np
from ex021_Brine_Feedback2D.modflow_model import CASE_DIR, ChemistryConfig, Config, build_model
from example_utils import (
    atomic_save,
    atomic_write_text,
    configure_logging,
    file_digest,
    library_path,
    runtime_path,
)

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    ChemistryOptions,
    FeedbackOptions,
    OutputOptions,
    SimulationConfig,
)
from mf6pqc.backends import initialize_modflow6
from mf6pqc.coupling.common import (
    build_standard_state,
    cache_basic_geometry,
    enforce_component_domains,
    finalize_results,
    get_calculated_density,
    get_coupling_time_step,
    read_concentrations_from_modflow,
    run_reaction_step,
    save_time_step_results,
    solve_modflow_solutions,
    update_selected_output,
    write_concentrations_to_modflow,
)
from mf6pqc.feedback import update_medium_properties, write_conductivity_for_step
from mf6pqc.permeability import BasePermeabilityUpdater, KozenyCarmanUpdater

MINERAL_MOLAR_VOLUMES = dict(
    Halite=0.0271, Carnallite=0.1737, Sylvite=0.0375, Gypsum=0.0739, Polyhalite=0.218
)
POROSITY = 0.2
MINERAL_VOLUME_FRACTIONS = dict(
    Halite=0.4, Carnallite=0.16, Sylvite=0.015, Gypsum=0.025, Polyhalite=0.04
)
INERT_FRACTION = 1 - POROSITY - sum(MINERAL_VOLUME_FRACTIONS.values())
ELEMENTS = ["Na", "K", "Mg", "Ca", "Cl", "S(6)", "C(4)"]
SCENARIOS = {
    "S00": "Fixed density and structure",
    "S10": "Density only",
    "S01": "Structure only",
    "S11": "Density and structure",
}
DEFAULT_CONFIG = Config(
    nx=200,
    nz=50,
    length=400.0,
    height=100.0,
    width=1.0,
    days=2400.0,
    dt=4.0,
    save_every=100.0,
    kv_ratio=0.3,
    injection_rate=5.0,
    injection_depth=10.0,
    extraction_depth=20.0,
    screen_conductance=100.0,
    outlet_head=110.0,
    alpha_l=2.0,
    alpha_t=0.2,
    diffusion=8.64e-05,
)
CHEMISTRY = ChemistryConfig(
    porosity=POROSITY,
    mineral_molar_volumes=MINERAL_MOLAR_VOLUMES,
    mineral_volume_fractions=MINERAL_VOLUME_FRACTIONS,
    elements=ELEMENTS,
    temperature=25.0,
    pressure=1.0,
    density=1.28,
)


def write_input(*, chemistry: ChemistryConfig) -> Path:
    """Write the case chemistry only when its generated input has changed."""
    if chemistry.inert_fraction <= 0:
        raise ValueError("Mineral fractions must leave a positive inert fraction")
    text = """TITLE Halite-saturated injection into a mixed evaporite assemblage
SOLUTION 90
    temp 25
    pH 6.818060
    units mol/L
    density 1.277848
    Cl 8.023463 charge
    K 0.444199
    Na 0.387030
    Ca 0.011049
    Mg 3.637958
    S(6) 0.051852
    C(4) 0.000088
EQUILIBRIUM_PHASES 90
    Halite 0 10
    Carnallite 0 10
    Sylvite 0 10
    Gypsum 0 10
    Polyhalite 0 10
SAVE solution 0
END
SOLUTION 91
    temp 25
    pH 7.314977
    units mol/L
    density 1.198844
    Cl 5.189443 charge
    K 0.000041
    Na 5.328907
    Ca 0.000031
    Mg 0.000515
    S(6) 0.070299
    C(4) 0.000088
EQUILIBRIUM_PHASES 91
    Halite 0 10
SAVE solution 1
END
EQUILIBRIUM_PHASES 1
"""
    for name, fraction in chemistry.mineral_volume_fractions.items():
        text += f"    {name} 0 {fraction / chemistry.mineral_molar_volumes[name]:.14g}\n"
    text += "END\nSELECTED_OUTPUT 1\n    -reset false\n    -high_precision true\nUSER_PUNCH 1\n"
    headings = (
        chemistry.elements
        + list(chemistry.mineral_molar_volumes)
        + ["d_" + m for m in chemistry.mineral_molar_volumes]
        + ["SI_" + m for m in chemistry.mineral_molar_volumes]
        + ["water_kg", "solution_L", "pH", "RHO"]
    )
    text += "    -headings " + " ".join(headings) + "\n    -start\n"
    expressions = [f'TOT("{e}")*TOT("water")/SOLN_VOL' for e in chemistry.elements]
    expressions += [f'EQUI("{m}")' for m in chemistry.mineral_molar_volumes]
    expressions += [f'EQUI_DELTA("{m}")' for m in chemistry.mineral_molar_volumes]
    expressions += [f'SI("{m}")' for m in chemistry.mineral_molar_volumes]
    expressions += ['TOT("water")', "SOLN_VOL", '-LA("H+")', "RHO"]
    for i, expr in enumerate(expressions, 1):
        text += f"    {i * 10} PUNCH {expr}\n"
    text += """    -end
END
KNOBS
    -iterations 400
    -step_size 10
    -diagonal_scale true
    -tolerance 1e-11
END
"""
    path = CASE_DIR / "input_data/input.pqi"
    if not path.exists() or path.read_text(encoding="utf-8") != text:
        path.write_text(text, encoding="utf-8")
    return path


def create_simulator(
    *,
    nxyz: int,
    nthreads: int,
    scenario: str,
    label: str,
    chemistry: ChemistryConfig,
    save_steps: list[int] | None = None,
    progress_interval: int = 1000,
    k33_ratio: float = 0.6,
    permeability_updater: BasePermeabilityUpdater | None = None,
) -> MF6PQC:
    """Create the simulator with the scenario's density and structure feedback settings."""
    if len(scenario) != 3 or scenario[0] != "S" or any(flag not in "01" for flag in scenario[1:]):
        raise ValueError(f"Unknown scenario: {scenario}")
    workspace = runtime_path(__file__, "simulation") / label
    output_dir = runtime_path(__file__, "output") / label
    simulation_config = SimulationConfig(
        case_name="ex021",
        nxyz=nxyz,
        nthreads=nthreads,
        paths=BackendPaths(
            database=CASE_DIR / "input_data" / "database.dat",
            chemistry_input=CASE_DIR / "input_data" / "input.pqi",
            modflow_library=library_path(),
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=chemistry.temperature,
            pressure_atm=chemistry.pressure,
            porosity=chemistry.porosity,
            saturation=1,
            density_kg_per_litre=chemistry.density,
        ),
        chemistry=ChemistryOptions(
            print_chemistry_mask=0,
            transport_water_component=False,
            use_solution_density_volume=False,
        ),
        feedback=FeedbackOptions(
            update_density=scenario[1] == "1",
            use_phreeqc_calculated_density=True,
            update_porosity_and_k=scenario[2] == "1",
            mineral_molar_volumes=chemistry.mineral_molar_volumes,
            vertical_to_horizontal_k_ratio=k33_ratio,
            permeability_updater=permeability_updater,
        ),
        output=OutputOptions(save_steps=save_steps, progress_interval=progress_interval),
        fail_on_modflow_nonconvergence=True,
    )
    return MF6PQC.from_config(simulation_config)


def simulation_fingerprint() -> dict:
    """Hash inputs and source files to recognize reusable completed runs."""
    sources = ("run.py", "modflow_model.py")
    return {
        "database_sha256": file_digest(CASE_DIR / "input_data" / "database.dat"),
        "input_sha256": file_digest(CASE_DIR / "input_data" / "input.pqi"),
        "source_sha256": {
            **{name: file_digest(CASE_DIR / name) for name in sources},
            "example_utils.py": file_digest(CASE_DIR.parent / "example_utils.py"),
        },
    }


AUDIT_ELEMENTS = ["Na", "K", "Mg", "Ca", "Cl", "S"]
STOICH = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 1, 0, 2],
        [0, 1, 0, 0, 1],
        [0, 0, 0, 1, 2],
        [1, 3, 1, 0, 0],
        [0, 0, 0, 1, 4],
    ],
    dtype=float,
)


def inventories(sim, concentration, config, *, chemistry: ChemistryConfig):
    aq = concentration.reshape(sim.ncomps, config.nxyz)
    rows = [sim.components.index(e) for e in AUDIT_ELEMENTS]
    solid = np.stack(
        [sim.selected_output[sim.headings.index(m)] for m in chemistry.mineral_molar_volumes]
    )
    return (
        ((aq[rows] * sim.porosity).sum(axis=1) + STOICH @ solid.sum(axis=1))
        * config.cell_volume
        * 1000
    )


def run_scenario(
    config: Config,
    scenario: str,
    label: str,
    nthreads: int,
    legacy_commit: bool = False,
    *,
    chemistry: ChemistryConfig,
    description: str,
) -> None:
    save_steps = frozenset(config.save_steps)
    out = runtime_path(__file__, "output") / label
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    k = config.k_field()
    completed = False
    atomic_write_text(out / "status.json", json.dumps(dict(completed=False, state="running")))
    try:
        with create_simulator(
            nxyz=config.nxyz,
            nthreads=nthreads,
            scenario=scenario,
            label=label,
            save_steps=config.save_steps,
            progress_interval=25,
            k33_ratio=config.kv_ratio,
            permeability_updater=KozenyCarmanUpdater(),
            chemistry=chemistry,
        ) as sim:
            initial = sim.setup({"solution": 0, "equilibrium_phases": 1})
            injection = sim.get_initial_concentrations(1)
            j = sim.components.index("Charge")
            initial[j * config.nxyz : (j + 1) * config.nxyz] = 0.0
            injection[j] = 0.0
            rho0 = float(sim.phreeqc_rm.GetDensityCalculated()[0]) * 1000
            build_model(
                workspace=sim.workspace,
                species=sim.components,
                initial_concentrations=initial,
                inflow_concentrations=injection,
                config=config,
                porosity=chemistry.porosity,
                hydraulic_conductivity=k,
                reference_density=rho0,
                update_density=scenario[1] == "1",
            )
            initialize_modflow6(sim)
            cache_basic_geometry(sim)
            state = build_standard_state(sim)
            api = sim.modflow_api
            varnames = list(api.get_input_var_names())
            (out / "api_variables.txt").write_text("\n".join(varnames), encoding="utf-8")
            qout = api.get_value_ptr(api.get_var_address("SIMVALS", "gwf_model", "OUTLET"))
            zout = config.outlet_layers * config.nx + config.nx - 1
            initial_aq = initial.reshape(sim.ncomps, config.nxyz).copy()
            mass0 = inventories(sim, initial, config, chemistry=chemistry)
            net_flux = np.zeros(6)
            erows = [sim.components.index(e) for e in AUDIT_ELEMENTS]
            heads = []
            flux_out = []
            aq_frames = [initial_aq.copy()]
            history = []
            (out / "metadata.json").write_text(
                json.dumps(
                    dict(
                        config=config.to_dict(),
                        chemistry=chemistry.to_dict(),
                        scenario=scenario,
                        label=label,
                        description=description,
                        phi0=chemistry.porosity,
                        inert_fraction=chemistry.inert_fraction,
                        rho_reference_kg_m3=rho0,
                        components=sim.components,
                        mineral_order=list(chemistry.mineral_molar_volumes),
                        initial_moles=mass0.tolist(),
                        injection_mol_L=injection.tolist(),
                        conservative_porosity_commit=not legacy_commit,
                        k_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
                        **simulation_fingerprint(),
                    ),
                    indent=2,
                ),
                encoding="utf-8",
            )
            np.save(out / "initial_K.npy", k)
            while state.logical_step < config.steps:
                dt = get_coupling_time_step(state)
                api.prepare_time_step(dt)
                write_conductivity_for_step(sim, state.current_k11, state.logical_step)
                density = get_calculated_density(sim) if sim.if_update_density else None
                solve_modflow_solutions(sim, state, density)
                api.finalize_time_step()
                state.current_time = float(api.get_current_time())
                read_concentrations_from_modflow(
                    state.concentration_variables, state.species_slices, state.transported
                )
                trans = state.transported.reshape(sim.ncomps, config.nxyz)
                flux = config.injection_rate * injection[erows]
                flux += np.sum(
                    np.maximum(qout, 0)[None, :] * initial_aq[erows][:, zout]
                    + np.minimum(qout, 0)[None, :] * trans[erows][:, zout],
                    axis=1,
                )
                net_flux += flux * dt * 1000
                enforce_component_domains(
                    state.transported, sim.components, state.species_slices, sim.signed_components
                )
                run_reaction_step(
                    sim, state.transported, state.reacted, state.last_reaction_time, dt
                )
                state.last_reaction_time = state.current_time
                update_selected_output(sim)
                state.current_k11 = update_medium_properties(
                    sim, state.current_k11, state.logical_step
                )
                if sim.if_update_porosity_K and (not legacy_commit):
                    state.reacted[:] = sim.phreeqc_rm.GetConcentrations()
                write_concentrations_to_modflow(
                    state.concentration_variables, state.species_slices, state.reacted
                )
                save_time_step_results(sim, state.logical_step, state.current_time)
                aq = state.reacted.reshape(sim.ncomps, config.nxyz)
                current = inventories(sim, state.reacted, config, chemistry=chemistry)
                error = current - mass0 - net_flux
                mineral = sim.selected_output[sim.headings.index("Carnallite")]
                car0 = sim.results[0][sim.headings.index("Carnallite")]
                dissolved = np.maximum(car0 - mineral, 0)
                log = dict(
                    day=state.current_time,
                    wall_s=time.perf_counter() - start,
                    min_phi=float(sim.porosity.min()),
                    max_phi=float(sim.porosity.max()),
                    max_k_ratio=(
                        float(np.max(state.current_k11 / k))
                        if state.current_k11 is not None
                        else 1.0
                    ),
                    net_outflow=float(-qout.sum()),
                    outward_flow=float(-np.minimum(qout, 0).sum()),
                    backflow=float(np.maximum(qout, 0).sum()),
                    carnallite_dissolved_mol=float(dissolved.sum() * 1000 * config.cell_volume),
                    carnallite_dissolution_centroid_z=float(
                        np.sum(dissolved.reshape(config.nz, config.nx) * config.z[:, None])
                        / max(dissolved.sum(), 1e-30)
                    ),
                    outlet_K_mol_L=float(
                        np.sum(-np.minimum(qout, 0) * aq[sim.components.index("K"), zout])
                        / max(-np.minimum(qout, 0).sum(), 1e-30)
                    ),
                    max_mass_error_fraction=float(np.max(np.abs(error) / mass0)),
                    mass_error_fraction=(error / mass0).tolist(),
                )
                history.append(log)
                step = state.logical_step + 1
                if step in save_steps:
                    heads.append(api.get_value(api.get_var_address("X", "gwf_model")).copy())
                    flux_out.append(qout.copy())
                    aq_frames.append(aq.copy())
                    checkpoint = out / "checkpoints"
                    checkpoint.mkdir(exist_ok=True)
                    atomic_save(checkpoint / f"selected_{step:05d}.npy", sim.selected_output)
                    atomic_save(checkpoint / f"aqueous_{step:05d}.npy", aq)
                    atomic_save(checkpoint / f"porosity_{step:05d}.npy", sim.porosity)
                    atomic_save(
                        checkpoint / f"K_{step:05d}.npy",
                        state.current_k11 if state.current_k11 is not None else k,
                    )
                    atomic_write_text(out / "history.partial.json", json.dumps(history, indent=2))
                if step == 1 or step % 25 == 0:
                    print(json.dumps(dict(step=step, **log)), flush=True)
                    (out / "progress.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
                state.logical_step += 1
            finalize_results(sim, state.logical_step, start)
            sim.save_results()
            np.save(out / "aqueous_transport.npy", np.stack(aq_frames))
            np.save(out / "heads.npy", np.stack(heads))
            np.save(out / "outlet_flux.npy", np.stack(flux_out))
            (out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            if not sim.if_update_porosity_K:
                np.save(
                    out / "results_porosity.npy",
                    np.full((len(aq_frames), config.nxyz), chemistry.porosity),
                )
                np.save(out / "results_K.npy", np.tile(k, (len(aq_frames), 1)))
        completed = True
    finally:
        atomic_write_text(
            out / "status.json",
            json.dumps(
                {"completed": completed, "wall_seconds": time.perf_counter() - start}, indent=2
            ),
        )


def main() -> None:
    """Parse the command line and run or reuse the requested feedback scenarios."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    parser.add_argument("--nx", type=int, default=DEFAULT_CONFIG.nx)
    parser.add_argument("--nz", type=int, default=DEFAULT_CONFIG.nz)
    parser.add_argument("--days", type=float, default=DEFAULT_CONFIG.days)
    parser.add_argument("--dt", type=float, default=DEFAULT_CONFIG.dt)
    parser.add_argument("--save-every", type=float, default=DEFAULT_CONFIG.save_every)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--label")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--legacy-commit", action="store_true", help="Audit uncorrected pore-volume handoff only"
    )
    args = parser.parse_args()
    config = replace(
        DEFAULT_CONFIG,
        nx=args.nx,
        nz=args.nz,
        days=args.days,
        dt=args.dt,
        save_every=args.save_every,
    )
    if args.threads <= 0:
        parser.error("--threads must be positive")
    write_input(chemistry=CHEMISTRY)
    if args.scenario == "all":
        if args.label:
            parser.error("--label requires a single --scenario")
        for scenario in SCENARIOS:
            command = [
                sys.executable,
                "-B",
                str(CASE_DIR / "run.py"),
                "--scenario",
                scenario,
                "--nx",
                str(args.nx),
                "--nz",
                str(args.nz),
                "--days",
                str(args.days),
                "--dt",
                str(args.dt),
                "--save-every",
                str(args.save_every),
                "--threads",
                str(args.threads),
            ]
            if args.legacy_commit:
                command.append("--legacy-commit")
            if args.force:
                command.append("--force")
            subprocess.run(command, cwd=CASE_DIR, check=True)
        return
    label = args.label or args.scenario + "_shallow"
    output_dir = runtime_path(__file__, "output") / label
    status_path = output_dir / "status.json"
    if (
        not args.force
        and status_path.exists()
        and json.loads(status_path.read_text(encoding="utf-8")).get("completed")
    ):
        saved = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
        active_config = config.to_dict()
        fingerprint = simulation_fingerprint()
        if (
            any((saved["config"].get(key) != value for key, value in active_config.items()))
            or saved["scenario"] != args.scenario
            or saved.get("chemistry") != CHEMISTRY.to_dict()
            or any((saved.get(key) != value for key, value in fingerprint.items()))
            or (saved.get("conservative_porosity_commit", True) != (not args.legacy_commit))
        ):
            raise ValueError(
                "Existing label has different settings; choose a new --label or use --force."
            )
        required = (
            "results.npy",
            "results_times.npy",
            "results_headings.txt",
            "results_porosity.npy",
            "results_K.npy",
            "aqueous_transport.npy",
            "heads.npy",
            "outlet_flux.npy",
            "history.json",
            "initial_K.npy",
        )
        if any(not (output_dir / name).is_file() for name in required):
            raise ValueError("Completed results are incomplete; use --force")
        np.testing.assert_array_equal(
            np.round(np.load(output_dir / "results_K.npy", mmap_mode="r")[0], 8),
            np.round(config.k_field(), 8),
            err_msg="Initial conductivity differs; choose a new --label or use --force.",
        )
        print(f"{label}: using completed results. Use --force for a fresh run.")
        return
    run_scenario(
        config,
        args.scenario,
        label,
        args.threads,
        args.legacy_commit,
        chemistry=CHEMISTRY,
        description=SCENARIOS[args.scenario],
    )


if __name__ == "__main__":
    configure_logging()
    main()
