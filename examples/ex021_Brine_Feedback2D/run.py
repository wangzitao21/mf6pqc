"""Run the brine density and structure feedback scenarios."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CASE_DIR.parents[1]))
import numpy as np

from examples.ex021_Brine_Feedback2D.modflow_model import ChemistryConfig, Config, build_model
from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    CouplingHooks,
    FeedbackOptions,
    OutputOptions,
    ProcessBackendFactory,
    SimulationConfig,
)
from mf6pqc.permeability import BasePermeabilityUpdater, KozenyCarmanUpdater

INPUT_DIR = CASE_DIR / "input_data"
WORKSPACE = CASE_DIR / "simulation"
OUTPUT_DIR = CASE_DIR / "output"
MODFLOW_LIBRARY = Path(
    os.environ.get(
        "MF6PQC_LIBMF6",
        CASE_DIR.parents[1]
        / "bin"
        / "mf6.8.0"
        / {"win32": "libmf6.dll", "darwin": "libmf6.dylib"}.get(sys.platform, "libmf6.so"),
    )
)

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
    workspace = WORKSPACE / label
    output_dir = OUTPUT_DIR / label
    simulation_config = SimulationConfig(
        case_name="ex021",
        nxyz=nxyz,
        nthreads=1,
        backend_factory=ProcessBackendFactory(processes=min(nthreads, os.cpu_count() or 1)),
        paths=BackendPaths(
            database=CASE_DIR / "input_data" / "database.dat",
            chemistry_input=CASE_DIR / "input_data" / "input.pqi",
            modflow_library=MODFLOW_LIBRARY,
            workspace=workspace,
            output_directory=output_dir,
        ),
        fields=CellFields(
            temperature_c=chemistry.temperature,
            pressure_atm=chemistry.pressure,
            porosity=chemistry.porosity,
            density_kg_per_litre=chemistry.density,
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
    *,
    chemistry: ChemistryConfig,
    description: str,
) -> None:
    save_steps = frozenset(config.save_steps)
    out = OUTPUT_DIR / label
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    k = config.k_field()
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
        qout = None

        def on_initialize(sim, state):
            nonlocal qout
            api = sim.modflow_api
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
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        np.save(out / "initial_K.npy", k)

        def on_transport(sim, state, dt):
            trans = state.transported.reshape(sim.ncomps, config.nxyz)
            flux = config.injection_rate * injection[erows]
            flux += np.sum(
                np.maximum(qout, 0)[None, :] * initial_aq[erows][:, zout]
                + np.minimum(qout, 0)[None, :] * trans[erows][:, zout],
                axis=1,
            )
            net_flux[:] += flux * dt * 1000

        def on_step(sim, state):
            api = sim.modflow_api
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
                max_k_ratio=float(np.max(state.current_k11 / k))
                if state.current_k11 is not None
                else 1.0,
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
            step = state.logical_step
            if step in save_steps:
                heads.append(api.get_value(api.get_var_address("X", "gwf_model")).copy())
                flux_out.append(qout.copy())
                aq_frames.append(aq.copy())

        sim.run(
            hooks=CouplingHooks(
                on_initialize=on_initialize, on_transport=on_transport, on_step=on_step
            )
        )
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


def main() -> None:
    for scenario, description in SCENARIOS.items():
        run_scenario(
            DEFAULT_CONFIG,
            scenario,
            scenario + "_shallow",
            8,
            chemistry=CHEMISTRY,
            description=description,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
