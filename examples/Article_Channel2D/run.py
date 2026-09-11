"""Run a scenario with MF6PQC and retain an explicit elemental mass audit.

The case-local commit corrects the concentration volume basis after porosity
changes: C_new = m_aqueous / (phi_new * representative volume). This uses the
public PhreeqcRM getter after MF6PQC sets porosity. The repository core is not
modified. The notebook describes the quasi-steady hydraulic approximation.
"""

import sys

sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import subprocess
import sys
import time

import numpy as np
from modflow_model import (
    CASE,
    INERT,
    PHI,
    SCENARIOS,
    VM,
    Config,
    build_model,
    make_simulator,
    write_input,
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
from mf6pqc.output_processing import _atomic_save, _atomic_write_text
from mf6pqc.permeability import KozenyCarmanUpdater

# Stoichiometry of non-water elements; hydrous minerals contribute H/O too,
# but the main mass audit intentionally covers the six salt-forming elements.
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


def inventories(sim, concentration, c):
    aq = concentration.reshape(sim.ncomps, c.nxyz)
    rows = [sim.components.index(e) for e in AUDIT_ELEMENTS]
    solid = np.stack([sim.selected_output[sim.headings.index(m)] for m in VM])
    return (
        ((aq[rows] * sim.porosity).sum(axis=1) + STOICH @ solid.sum(axis=1)) * c.cell_volume * 1000
    )


def run(c, scenario, label, threads, legacy_commit=False):
    out = CASE / "output" / label
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    k = c.k_field()
    sim = make_simulator(
        c.nxyz,
        threads,
        scenario=scenario,
        label=label,
        save_steps=c.save_steps,
        progress_interval=25,
        k33_ratio=c.kv_ratio,
        permeability_updater=KozenyCarmanUpdater(),
    )
    completed = False
    _atomic_write_text(out / "status.json", json.dumps(dict(completed=False, state="running")))
    try:
        initial = sim.setup({"solution": 0, "equilibrium_phases": 1})
        injection = sim.get_initial_concentrations(1)
        # Tiny charge residual of the fully balanced external solutions.
        j = sim.components.index("Charge")
        initial[j * c.nxyz : (j + 1) * c.nxyz] = 0.0
        injection[j] = 0.0
        rho0 = float(sim.phreeqc_rm.GetDensityCalculated()[0]) * 1000
        build_model(
            c, sim.workspace, sim.components, initial, injection, k, scenario[1] == "1", rho0
        )
        initialize_modflow6(sim)
        cache_basic_geometry(sim)
        state = build_standard_state(sim)
        api = sim.modflow_api
        varnames = list(api.get_input_var_names())
        (out / "api_variables.txt").write_text("\n".join(varnames), encoding="utf-8")
        qout = api.get_value_ptr(api.get_var_address("SIMVALS", "gwf_model", "OUTLET"))
        zout = c.outlet_layers * c.nx + c.nx - 1
        initial_aq = initial.reshape(sim.ncomps, c.nxyz).copy()
        mass0 = inventories(sim, initial, c)
        net_flux = np.zeros(6)
        erows = [sim.components.index(e) for e in AUDIT_ELEMENTS]
        heads = []
        flux_out = []
        aq_frames = [initial_aq.copy()]
        history = []
        (out / "metadata.json").write_text(
            json.dumps(
                dict(
                    config=c.to_dict(),
                    scenario=scenario,
                    label=label,
                    description=SCENARIOS[scenario],
                    phi0=PHI,
                    inert_fraction=INERT,
                    rho_reference_kg_m3=rho0,
                    components=sim.components,
                    mineral_order=list(VM),
                    initial_moles=mass0.tolist(),
                    injection_mol_L=injection.tolist(),
                    conservative_porosity_commit=not legacy_commit,
                    database_sha256=hashlib.sha256(
                        (CASE / "input_data/pitzer.dat").read_bytes()
                    ).hexdigest(),
                    input_sha256=hashlib.sha256(
                        (CASE / "input_data/input.pqi").read_bytes()
                    ).hexdigest(),
                    k_sha256=hashlib.sha256(k.tobytes()).hexdigest(),
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(out / "initial_K.npy", k)
        while state.logical_step < c.steps:
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
            trans = state.transported.reshape(sim.ncomps, c.nxyz)
            # GWT uses the transported (pre-reaction) implicit endpoint at sinks.
            flux = c.injection_rate * injection[erows]
            flux += np.sum(
                np.maximum(qout, 0)[None, :] * initial_aq[erows][:, zout]
                + np.minimum(qout, 0)[None, :] * trans[erows][:, zout],
                axis=1,
            )
            net_flux += flux * dt * 1000
            enforce_component_domains(
                state.transported, sim.components, state.species_slices, sim.signed_components
            )
            run_reaction_step(sim, state.transported, state.reacted, state.last_reaction_time, dt)
            state.last_reaction_time = state.current_time
            update_selected_output(sim)
            state.current_k11 = update_medium_properties(sim, state.current_k11, state.logical_step)
            # Re-query AFTER SetPorosity to conserve pore-water component moles.
            if sim.if_update_porosity_K and not legacy_commit:
                state.reacted[:] = sim.phreeqc_rm.GetConcentrations()
            write_concentrations_to_modflow(
                state.concentration_variables, state.species_slices, state.reacted
            )
            save_time_step_results(sim, state.logical_step, state.current_time)
            aq = state.reacted.reshape(sim.ncomps, c.nxyz)
            current = inventories(sim, state.reacted, c)
            error = current - mass0 - net_flux
            mineral = sim.selected_output[sim.headings.index("Carnallite")]
            car0 = sim.results[0][sim.headings.index("Carnallite")]
            dissolved = np.maximum(car0 - mineral, 0)
            log = dict(
                day=state.current_time,
                wall_s=time.perf_counter() - start,
                min_phi=float(sim.porosity.min()),
                max_phi=float(sim.porosity.max()),
                max_k_ratio=float(np.max(state.current_k11 / k))
                if state.current_k11 is not None
                else 1.0,
                net_outflow=float(-qout.sum()),
                outward_flow=float(-np.minimum(qout, 0).sum()),
                backflow=float(np.maximum(qout, 0).sum()),
                carnallite_dissolved_mol=float(dissolved.sum() * 1000 * c.cell_volume),
                carnallite_dissolution_centroid_z=float(
                    np.sum(dissolved.reshape(c.nz, c.nx) * c.z[:, None])
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
            if step in c.save_steps:
                heads.append(api.get_value(api.get_var_address("X", "gwf_model")).copy())
                flux_out.append(qout.copy())
                aq_frames.append(aq.copy())
                # Durable analysis checkpoints. These are output snapshots, not
                # complete solver restart states; preserve every finished frame.
                checkpoint = out / "checkpoints"
                checkpoint.mkdir(exist_ok=True)
                _atomic_save(checkpoint / f"selected_{step:05d}.npy", sim.selected_output)
                _atomic_save(checkpoint / f"aqueous_{step:05d}.npy", aq)
                _atomic_save(checkpoint / f"porosity_{step:05d}.npy", sim.porosity)
                _atomic_save(
                    checkpoint / f"K_{step:05d}.npy",
                    state.current_k11 if state.current_k11 is not None else k,
                )
                _atomic_write_text(out / "history.partial.json", json.dumps(history, indent=2))
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
            np.save(out / "results_porosity.npy", np.full((len(aq_frames), c.nxyz), PHI))
            np.save(out / "results_K.npy", np.tile(k, (len(aq_frames), 1)))
        completed = True
    finally:
        sim.finalize()
        (out / "status.json").write_text(
            json.dumps(
                dict(completed=completed, wall_seconds=time.perf_counter() - start), indent=2
            ),
            encoding="utf-8",
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["all", *SCENARIOS], default="all")
    p.add_argument("--nx", type=int, default=200)
    p.add_argument("--nz", type=int, default=50)
    p.add_argument("--days", type=float, default=2400)
    p.add_argument("--dt", type=float, default=4)
    p.add_argument("--save-every", type=float, default=100)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--label")
    p.add_argument(
        "--legacy-commit", action="store_true", help="Audit uncorrected pore-volume handoff only"
    )
    a = p.parse_args()
    c = Config(nx=a.nx, nz=a.nz, days=a.days, dt=a.dt, save_every=a.save_every)
    if not np.isclose(c.steps * c.dt, c.days):
        raise ValueError("Duration must be an integer number of steps")
    if not (CASE / "input_data/input.pqi").exists():
        write_input()
    if a.scenario == "all":
        if a.label:
            p.error("--label requires a single --scenario")
        for scenario in SCENARIOS:
            command = [
                sys.executable,
                "-B",
                str(CASE / "run.py"),
                "--scenario",
                scenario,
                "--nx",
                str(a.nx),
                "--nz",
                str(a.nz),
                "--days",
                str(a.days),
                "--dt",
                str(a.dt),
                "--save-every",
                str(a.save_every),
                "--threads",
                str(a.threads),
            ]
            if a.legacy_commit:
                command.append("--legacy-commit")
            subprocess.run(command, cwd=CASE, check=True)
        return
    label = a.label or (a.scenario + "_shallow")
    status = CASE / "output" / label / "status.json"
    if status.exists() and json.loads(status.read_text()).get("completed"):
        saved = json.loads((status.parent / "metadata.json").read_text())
        # Older metadata also records parameters used to generate the fixed
        # conductivity source. Compare active settings and verify that field
        # directly, so archived and current metadata schemas remain compatible.
        active_config = c.to_dict()
        if (
            any(saved["config"].get(key) != value for key, value in active_config.items())
            or saved["scenario"] != a.scenario
            or saved.get("conservative_porosity_commit", True) != (not a.legacy_commit)
        ):
            raise ValueError("Existing label has different settings; choose a new --label.")
        # NPF inputs retain eight decimal places. Feedback scenarios read that
        # written field, while fixed-property outputs retain the source precision.
        np.testing.assert_array_equal(
            np.round(np.load(status.parent / "results_K.npy", mmap_mode="r")[0], 8),
            np.round(c.k_field(), 8),
            err_msg="Initial conductivity differs; choose a new --label.",
        )
        print(f"{label}: using completed results. Choose a new --label for a fresh run.")
        return
    run(c, a.scenario, label, a.threads, a.legacy_commit)


if __name__ == "__main__":
    main()
