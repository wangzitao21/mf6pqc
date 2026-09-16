"""Implicit mineral--transport coupling with PhreeqcRM speciation.

The nonlinear unknowns are transformed mineral inventories. Physical fields
are predicted once per step and committed after convergence, outside Newton.
"""

from __future__ import annotations

import time
from types import MappingProxyType

import numpy as np

from mf6pqc.backends import initialize_modflow6
from mf6pqc.constants import MAX_POROSITY, MIN_POROSITY, SECONDS_PER_DAY
from mf6pqc.coupling.common import (
    advance_to_end,
    build_standard_state,
    cache_basic_geometry,
    finalize_results,
    get_coupling_time_step,
    read_concentrations_from_modflow,
    save_time_step_results,
    should_save_time_step,
    update_selected_output,
    validate_setup,
)
from mf6pqc.exceptions import ConfigurationError, ConvergenceError, CouplingError
from mf6pqc.feedback import write_conductivity_for_step
from mf6pqc.kinetics import KineticState
from mf6pqc.properties import update_diffc
from mf6pqc.utils import ensure_array, get_gwt_model_name


def _readonly(values):
    result = values.view()
    result.flags.writeable = False
    return result


class _NativeTransport:
    def __init__(self, sim, state):
        self.api, self.sim, self.state = sim.modflow_api, sim, state
        self.n = sim.nxyz
        self.flow = self.scalar(sim.flow_model_name + "/IDSOLN")
        if np.any(self.ptr(sim.flow_model_name + "/" + sim.npf_package_name + "/ICELLTYPE") != 0):
            raise ConfigurationError(
                "Implicit requires confined (ICELLTYPE=0), fully saturated flow cells"
            )
        self.ids, self.old, self.sources, self.nodes, self.counts, self.csr = [], [], [], [], [], []
        self.ptrs = [v["ptr"] for v in state.concentration_variables.values()]
        self.ibounds = []
        for component in sim.components:
            model = get_gwt_model_name(component).upper()
            sid = self.scalar(model + "/IDSOLN")
            self.ids.append(sid)
            if not np.allclose(
                self.ptr(model + "/DIS/TOP"), sim.top_arr, rtol=0, atol=1e-12
            ) or not np.allclose(self.ptr(model + "/DIS/BOT"), sim.botm_arr, rtol=0, atol=1e-12):
                raise ConfigurationError("Implicit GWT geometry must match the GWF grid")
            if not np.allclose(self.ptr(model + "/MST/THETAM"), sim.porosity, rtol=0, atol=1e-12):
                raise ConfigurationError("Implicit chemistry and GWT initial porosities must match")
            self.old.append(self.ptr(model + "/XOLD"))
            self.ibounds.append(self.ptr(model + "/IBOUND"))
            self.sources.append(self.ptr(model + "/SRC/SMASSRATE"))
            self.nodes.append(self.ptr(model + "/SRC/NODELIST"))
            self.counts.append(self.ptr(model + "/SRC/NBOUND"))
            if self.sources[-1].size != self.n:
                raise ConfigurationError("Implicit requires SRC with one entry per active cell")
            if self.scalar(model + "/ADV/IADVWT") != 0:
                raise ConfigurationError(
                    "Implicit currently requires UPSTREAM advection; TVD transport corrections "
                    "have not passed coarse-step convergence validation. Use SNIA/SIA for TVD."
                )
            for variable in ("MST/ISRB", "MST/IDCY"):
                if self.scalar(model + "/" + variable) != 0:
                    raise ConfigurationError(
                        "Implicit requires GWT MST without sorption/decay; "
                        f"{model}/{variable} is enabled"
                    )
            prefix = f"SLN_{sid}"
            ia = self.ptr(prefix + "/IA").astype(int) - 1
            ja = self.ptr(prefix + "/JA").astype(int) - 1
            if len(ia) != self.n + 1 or len(ja) != ia[-1]:
                raise ConfigurationError("Implicit requires a separate IMS solution for each GWT")
            self.csr.append((self.ptr(prefix + "/AMAT"), ja, ia))
        if len(set([self.flow, *self.ids])) != len(self.ids) + 1:
            raise ConfigurationError("Implicit requires separate GWF and component IMS solutions")
        model = get_gwt_model_name(sim.components[0]).upper()
        self.volume = self.ptr(model + "/DIS/AREA").copy() * sim.cell_thick
        if self.volume.shape != (self.n,) or np.any(self.volume <= 0):
            raise ConfigurationError(
                "Implicit requires positive cell volumes on a fully active grid"
            )

    def ptr(self, name):
        return self.api.get_value_ptr(name.upper())

    def scalar(self, name):
        return int(self.ptr(name).ravel()[0])

    def solve(self, sid):
        # A sparse direct predictor also handles IMS stagnation for almost
        # neutral signed Charge. MF6 must still confirm convergence afterward.
        maximum = int(self.state.solution_iterations[sid][0])
        converged = False
        try:
            for count in range(1, maximum + 1):
                if self.api.solve(sid):
                    converged = True
                    break
                if count == 2 and sid in self.ids:
                    from scipy.sparse import csr_matrix
                    from scipy.sparse.linalg import spsolve

                    i = self.ids.index(sid)
                    values, ja, ia = self.csr[i]
                    matrix = csr_matrix((values.copy(), ja, ia), shape=(self.n, self.n))
                    rhs = self.ptr(f"SLN_{sid}/RHS").copy()
                    predictor = spsolve(matrix, rhs)
                    if not np.all(np.isfinite(predictor)):
                        raise ConvergenceError("Non-finite MODFLOW direct predictor")
                    self.ptrs[i][:] = predictor
                    d = self.sim.implicit_diagnostics
                    d["transport_direct_predictors"] = d.get("transport_direct_predictors", 0) + 1
        finally:
            self.api.finalize_solve(sid)
        if not converged:
            self.sim.modflow_convergence_failures.append(
                dict(solution_id=sid, iterations=count, time_days=self.state.current_time)
            )
            raise ConvergenceError(f"Implicit MODFLOW solution {sid} did not converge")

    def source(self, i, value):
        nodes = self.nodes[i].astype(int) - 1
        if self.counts[i][0] != self.n or not np.array_equal(np.sort(nodes), np.arange(self.n)):
            raise ConfigurationError("Implicit SRC must cover every active cell exactly once")
        self.sources[i][:] = value[nodes]

    def unreacted(self, old):
        self.api.prepare_solve(self.flow)
        self.solve(self.flow)
        for i, sid in enumerate(self.ids):
            free = self.ibounds[i] > 0
            self.ptrs[i][free] = old[i, free]
            self.api.prepare_solve(sid)
            if self.state.logical_step == 0 and np.any(self.sources[i]):
                raise ConfigurationError(
                    "Implicit reserves SRC for reactions; initialize its mass rates to zero"
                )
            self.source(i, np.zeros(self.n))
            # Near-zero signed charge is often charge-balance roundoff. Keep
            # its old field as the IMS initial guess to avoid BCGS breakdown.
            self.ptrs[i][free] = old[i, free] if np.max(abs(old[i])) < 1e-10 else 0.0
            self.solve(sid)
        base = np.array(self.ptrs)
        return base, self.matrices()

    def matrices(self):
        from scipy.sparse import csr_matrix

        return [csr_matrix((v.copy(), ja, ia), shape=(self.n, self.n)) for v, ja, ia in self.csr]

    def commit(self, old, concentrations, rates):
        for i, sid in enumerate(self.ids):
            if not np.any(rates[i]):
                continue
            free = self.ibounds[i] > 0
            self.ptrs[i][free] = old[i, free]
            self.api.prepare_solve(sid)
            self.old[i][:] = old[i]
            self.source(i, rates[i] * self.volume)
            self.ptrs[i][:] = concentrations[i]
            self.solve(sid)
        return np.array(self.ptrs)


def run_implicit(sim):
    validate_setup(sim)
    options = sim.config.implicit
    reactions = options.reactions
    if not reactions:
        raise ConfigurationError("Implicit requires ImplicitOptions(reactions=...)")
    if (
        sim.componentH2O
        or sim.solution_density_volume
        or sim.if_update_density
        or sim.energy_enabled
    ):
        raise ConfigurationError(
            "Implicit currently requires total H/O, user solution volume, fixed density and temperature"
        )
    if not np.all(sim.saturation == 1) or sim.has_water_only_sinks:
        raise ConfigurationError(
            "Implicit requires saturated cells without water-only sink compensation"
        )
    if getattr(sim.backend_factory, "processes", 1) != 1:
        raise ConfigurationError(
            "Implicit currently uses NativeBackendFactory (native threads are supported)"
        )
    unsupported = set(sim.initial_condition_modules) - {"solution", "kinetics"}
    if unsupported:
        raise ConfigurationError(
            "Implicit does not yet include additional immobile equilibrium inventories: "
            + ", ".join(sorted(unsupported))
        )
    logarithmic = all(
        r.saturation_index_heading is not None and r.driving_force is None for r in reactions
    )
    names = [r.name for r in reactions]
    mineral_rows, delta_rows, sr_rows = [], [], []
    for r in reactions:
        for heading in (r.name, "d_" + r.name):
            if sim.headings.count(heading) != 1:
                raise ConfigurationError(f"Implicit requires selected-output heading {heading!r}")
        mineral_rows.append(sim.headings.index(r.name))
        delta_rows.append(sim.headings.index("d_" + r.name))
        sr_heading = (
            r.saturation_index_heading if logarithmic else (r.saturation_heading or "SR_" + r.name)
        )
        if r.driving_force is None and sim.headings.count(sr_heading) != 1:
            raise ConfigurationError(f"Implicit requires saturation-ratio heading {sr_heading!r}")
        sr_rows.append(sim.headings.index(sr_heading) if r.driving_force is None else -1)
        unknown = set(r.stoichiometry) - set(sim.components)
        if unknown:
            raise ConfigurationError(f"Unknown transported components for {r.name}: {unknown}")
    if sim.phreeqc_rm.GetChemistryCellCount() != sim.nxyz:
        raise ConfigurationError("Implicit requires one chemistry cell per transport cell")
    native_names = set(sim.phreeqc_rm.GetKineticReactions())
    if native_names != set(names):
        raise ConfigurationError(
            f"Implicit reactions must cover the PHREEQC kinetic network: {sorted(native_names)}"
        )
    mineral = sim.selected_output[mineral_rows].copy()
    if np.any(mineral < 0):
        raise ConfigurationError("Initial mineral inventories must be nonnegative")
    minimum_amounts = np.array(
        [ensure_array(sim.nxyz, "minimum_amount", r.minimum_amount) for r in reactions]
    )
    if np.any(mineral < minimum_amounts * (1 - 1e-12)):
        raise ConfigurationError("Initial minerals must be at least their declared minimum_amount")
    minimum_u = minimum_amounts ** (1 - np.array([r.surface_exponent for r in reactions])[:, None])
    initial_m = mineral.copy()
    exponent = np.array([r.surface_exponent for r in reactions])[:, None]
    reference = np.array(
        [
            initial_m[i]
            if r.reference_amount is None
            else ensure_array(sim.nxyz, "reference_amount", r.reference_amount)
            for i, r in enumerate(reactions)
        ]
    )
    if np.any((reference <= 0) & (exponent > 0)):
        raise ConfigurationError(
            "Positive surface_exponent requires positive reference_amount in every cell"
        )
    reference = np.where(exponent == 0, 1, reference)
    rate_constants = np.array(
        [ensure_array(sim.nxyz, "rate_constant", r.rate_constant) for r in reactions]
    )
    a = (1 - exponent) * rate_constants / reference**exponent
    nu = np.array([[r.stoichiometry.get(c, 0.0) for r in reactions] for c in sim.components])
    nonnegative = np.array([c.casefold() not in sim.signed_components for c in sim.components])
    volumes = np.zeros(len(reactions))
    if sim.if_update_porosity_K:
        if set(sim.d_mineral_names) != set(names):
            raise ConfigurationError(
                "All porosity-changing mineral outputs must belong to the implicit network"
            )
        try:
            volumes = np.array([sim.mineral_molar_volumes[name] for name in names])
        except KeyError as exc:
            raise ConfigurationError(f"Missing mineral molar volume: {exc}") from exc
    phi0 = sim.porosity.copy()
    initialize_modflow6(sim)
    start = time.perf_counter()
    cache_basic_geometry(sim)
    state = build_standard_state(sim)
    native = _NativeTransport(sim, state)
    rm, api = sim.phreeqc_rm, sim.modflow_api
    rm.SetRebalanceFraction(0.0)  # Trial cell masks must not trigger repeated worker migration.
    old_c = np.empty((sim.ncomps, sim.nxyz))
    previous_rate = np.zeros_like(mineral)
    cached_derivative = signature = None
    iterations = []
    custom = any(r.driving_force is not None for r in reactions)
    diagnostics = sim.implicit_diagnostics
    diagnostics.update(
        chemistry_calls=0,
        max_newton_residual=0.0,
        max_transport_closure=0.0,
        max_speciation_drift=0.0,
        max_volume_error=0.0,
        max_source_balance_error=0.0,
        reactions=names,
        logarithmic_saturation=logarithmic,
    )
    diagnostics["kinetics"] = [
        {
            "name": reaction.name,
            "stoichiometry": dict(reaction.stoichiometry),
            "surface_exponent": reaction.surface_exponent,
            "rate_constant_mol_bulk_per_day": rate_constants[i].tolist(),
            "reference_amount_mol_bulk": reference[i].tolist(),
            "minimum_amount_mol_bulk": minimum_amounts[i].tolist(),
            "saturation_index_heading": reaction.saturation_index_heading,
            "saturation_heading": reaction.saturation_heading or "SR_" + reaction.name,
            "custom_driving_force": None
            if reaction.driving_force is None
            else (
                getattr(reaction.driving_force, "__module__", "")
                + "."
                + getattr(
                    reaction.driving_force, "__qualname__", type(reaction.driving_force).__name__
                )
            ),
        }
        for i, reaction in enumerate(reactions)
    ]
    diagnostics["tolerances"] = {
        "absolute": options.absolute_tolerance,
        "relative": options.relative_tolerance,
        "concentration": options.concentration_tolerance,
        "maximum_iterations": options.maximum_iterations,
        "chemical_jacobian": options.chemical_jacobian,
        "porosity_coupling": options.porosity_coupling,
        "predict_porosity": options.predict_porosity,
        "directed_transport_blocks": options.directed_transport_blocks,
    }
    tangent = None
    cell_kernel = None
    if options.chemical_jacobian == "species":
        if not logarithmic:
            raise ConfigurationError(
                "Species tangents require saturation_index_heading for every reaction"
            )

        tangent = SpeciationTangent(rm, sim.components, nu)
    rm.SetTimeStep(0.0)

    def phi(amount):
        value = phi0 + volumes @ (initial_m - amount) * sim.porosity_update_mask
        if np.any((value < MIN_POROSITY) | (value > MAX_POROSITY)):
            raise CouplingError(
                "Implicit porosity left its physical bounds; no clipping is permitted"
            )
        return value

    def speciate(c, evaluation_phi, active):
        rm.SetPorosity(evaluation_phi)
        rm.SetSaturation(active.astype(float))
        rm.SetConcentrations(c.ravel())
        rm.RunCells()
        diagnostics["chemistry_calls"] += 1
        update_selected_output(sim)
        back = np.asarray(rm.GetConcentrations()).reshape(c.shape)
        drift = float(np.max(abs(back[:, active] - c[:, active]), initial=0))
        diagnostics["max_speciation_drift"] = max(diagnostics["max_speciation_drift"], drift)
        if drift > options.concentration_tolerance:
            raise CouplingError(
                f"Implicit speciation changed transported component inventory by {drift:g} mol/L; "
                "check redox component representation or unmodelled equilibrium reactions"
            )

    def step(sim, state):
        nonlocal mineral, previous_rate, cached_derivative, signature, cell_kernel
        dt = get_coupling_time_step(state)
        old_m, old_phi = mineral.copy(), sim.porosity.copy()
        read_concentrations_from_modflow(
            state.concentration_variables, state.species_slices, old_c.ravel()
        )
        predicted = np.maximum(old_m - previous_rate * dt * 0.5, 0)
        evaluation_phi = phi(predicted) if options.predict_porosity else old_phi
        native_old = (
            old_c * (old_phi / evaluation_phi)
            if options.porosity_coupling == "conservative"
            else old_c.copy()
        )
        for ptr in sim.thetam_ptrs.values():
            ptr[:] = evaluation_phi
        api.prepare_time_step(dt)
        if sim.if_update_porosity_K:
            evaluation_k = sim.perm_updater.update(state.current_k11, old_phi, evaluation_phi)
            write_conductivity_for_step(sim, evaluation_k, state.logical_step, force=True)
        if sim.if_update_diffc:
            diffusion = update_diffc(evaluation_phi, sim.d0)
            for ptr in sim.diffc_ptrs:
                ptr[:] = diffusion
        base, matrices = native.unreacted(native_old)
        # Fixed-concentration rows represent reservoirs. Reaction sources change
        # their boundary flux, not the prescribed aqueous concentration.
        source_masks = np.array(
            [
                native.ptr(get_gwt_model_name(component) + "/IBOUND") > 0
                for component in sim.components
            ]
        )
        transport = TransportResponse(
            matrices, native.volume, nu, options.dense_limit, source_masks=source_masks
        )
        new_signature = [(r, g, tuple(d)) for r, g, d in transport.directions]
        if signature != new_signature:
            cached_derivative = None
            signature = new_signature
        active = np.any((old_m > 0) | (exponent == 0), axis=0)
        rm.SetTime((state.current_time + dt) * SECONDS_PER_DAY)

        last_evaluation_c = last_evaluation_cells = None

        def evaluate(c, m, cells=None):
            nonlocal last_evaluation_c, last_evaluation_cells
            selected_cells = active.copy()
            if cells is not None:
                selected_cells[:] = False
                selected_cells[cells] = active[cells]
            if np.any(selected_cells) and not (
                np.array_equal(last_evaluation_c, c)
                and np.array_equal(last_evaluation_cells, selected_cells)
            ):
                speciate(c, evaluation_phi, selected_cells)
                last_evaluation_c, last_evaluation_cells = c.copy(), selected_cells.copy()
            context = None
            if custom:
                context = KineticState(
                    MappingProxyType(dict(zip(sim.components, _readonly(c), strict=True))),
                    MappingProxyType(
                        dict(zip(sim.headings, _readonly(sim.selected_output), strict=True))
                    ),
                    MappingProxyType(dict(zip(names, _readonly(m), strict=True))),
                    _readonly(sim.temperature),
                    state.current_time + dt,
                )
            drive = np.array(
                [
                    (
                        sim.selected_output[row] * np.log(10)
                        if logarithmic
                        else 1 - sim.selected_output[row]
                    )
                    if reaction.driving_force is None
                    else np.broadcast_to(reaction.driving_force(context), (sim.nxyz,))
                    for reaction, row in zip(reactions, sr_rows, strict=True)
                ]
            )
            drive[:, ~active] = 0.0
            if not np.all(np.isfinite(drive)):
                raise CouplingError("Implicit kinetic driving force is non-finite")
            return drive if cells is None else drive[:, cells]

        def at_cells(c, m, cells):
            nonlocal cell_kernel
            if len(cells) == 1 and tangent is not None:
                if cell_kernel is None:
                    cell_kernel = CellSpeciation(sim, nu, sr_rows)
                i = cells[0]
                return cell_kernel.evaluate(
                    c[:, i : i + 1], i, evaluation_phi[i], state.current_time + dt
                )
            return evaluate(c, m, cells)

        evaluate.at_cells = at_cells
        if tangent is not None:

            def jacobian_at_cells(c, m, cells, directions):
                at_cells(c, m, cells)
                if len(cells) == 1:
                    return cell_kernel.derivative(directions)
                return tangent.derivative(directions, sim.nxyz, cells)

            evaluate.jacobian_at_cells = jacobian_at_cells

            def jacobian(c, m):
                evaluate(c, m)
                return tangent.derivative(transport.directions, sim.nxyz)

            evaluate.jacobian = jacobian

            def component_jacobian(c, m, directions):
                evaluate(c, m)
                return tangent.derivative(directions, sim.nxyz)

            evaluate.component_jacobian = component_jacobian
        solver = solve_log_reactions if logarithmic else solve_reactions
        arguments = (
            base,
            old_m,
            previous_rate,
            dt,
            a,
            exponent,
            transport,
            evaluate,
            nonnegative,
            options,
        )
        result = None
        if logarithmic and options.directed_transport_blocks:
            result = solve_directed_reactions(*arguments, minimum_amounts=minimum_amounts)
        if result is None:
            result = solver(
                *arguments,
                cached_derivative=cached_derivative,
                custom=custom,
                minimum_amounts=minimum_amounts,
            )
        if getattr(evaluate, "directed_blocks_used", False):
            diagnostics["directed_block_steps"] = diagnostics.get("directed_block_steps", 0) + 1
        diagnostics["coordinate_sweeps"] = diagnostics.get("coordinate_sweeps", 0) + getattr(
            evaluate, "coordinate_sweeps", 0
        )
        diagnostics["block_sweeps"] = diagnostics.get("block_sweeps", 0) + getattr(
            evaluate, "block_sweeps", 0
        )
        diagnostics["augmented_iterations"] = diagnostics.get("augmented_iterations", 0) + getattr(
            evaluate, "augmented_iterations", 0
        )
        mineral, c, cached_derivative, count, residual = result
        if logarithmic:
            # Independently verify local-kernel and block solutions against the
            # complete native chemistry state and the original kinetic equation.
            checked_log_sr = evaluate(c, mineral)
            old_u = old_m ** (1 - exponent)
            target = np.maximum(
                minimum_u, old_u + dt * a * np.expm1(np.minimum(checked_log_sr, 700))
            )
            target[(old_m == 0) & (exponent > 0)] = 0
            checked_residual = mineral ** (1 - exponent) - target
            scale = options.absolute_tolerance + options.relative_tolerance * abs(old_u)
            if np.max(abs(checked_residual) / scale) > 1.01:
                raise CouplingError(
                    "Full PHREEQC verification of implicit kinetic residual failed: "
                    f"{np.max(abs(checked_residual)):g}"
                )
            residual = max(residual, float(np.max(abs(checked_residual))))
        iterations.append(count)
        previous_rate = (old_m - mineral) / dt
        component_rate = nu @ previous_rate
        transported = native.commit(native_old, c, component_rate)
        closure = float(np.max(abs(transported - c)))
        diagnostics["max_newton_residual"] = max(diagnostics["max_newton_residual"], residual)
        diagnostics["max_transport_closure"] = max(diagnostics["max_transport_closure"], closure)
        if closure > options.concentration_tolerance:
            raise CouplingError(
                f"Implicit native transport closure failed: {closure:g} mol/L; "
                "check concentration-dependent GWT boundary conditions or flux corrections"
            )
        for i, matrix in enumerate(native.matrices()):
            rhs = native.ptr(f"SLN_{native.ids[i]}/RHS")
            error = abs(matrix @ transported[i] - rhs)
            diagnostics["max_source_balance_error"] = max(
                diagnostics["max_source_balance_error"], float(np.max(error))
            )
            if np.any(
                error
                > options.concentration_tolerance * np.asarray(abs(matrix).sum(axis=1)).ravel()
            ):
                raise CouplingError("Implicit native transport equation balance failed")
        api.finalize_time_step()
        sim.porosity = phi(mineral)
        if sim.if_update_porosity_K:
            state.current_k11 = sim.perm_updater.update(state.current_k11, old_phi, sim.porosity)
        endpoint = (
            transported * (evaluation_phi / sim.porosity)
            if options.porosity_coupling == "conservative"
            else transported.copy()
        )
        endpoint[~source_masks] = transported[~source_masks]
        # Lagged porosity keeps concentration fixed and changes aqueous inventory.
        storage_change = sim.porosity * endpoint - evaluation_phi * transported
        diagnostics["max_property_storage_change_mol_bulk"] = max(
            diagnostics.get("max_property_storage_change_mol_bulk", 0.0),
            float(np.max(abs(storage_change[:, sim.porosity_update_mask]))),
        )
        rm.SetPorosity(sim.porosity)
        changed = np.flatnonzero(np.any(mineral != old_m, axis=0))
        if changed.size:
            lines = []
            for i in changed:
                lines.append(f"KINETICS_MODIFY {i}")
                for j, name in enumerate(names):
                    lines.append(f"-component {name}\n-m {mineral[j, i]:.17g}")
            rm.RunString(True, False, False, "\n".join(lines) + "\nEND\n")
        for i, ptr in enumerate(native.ptrs):
            ptr[:] = endpoint[i]
        for ptr in sim.thetam_ptrs.values():
            ptr[:] = sim.porosity
        if sim.if_update_diffc:
            sim.current_diffusion = update_diffc(sim.porosity, sim.d0)
            for ptr in sim.diffc_ptrs:
                ptr[:] = sim.current_diffusion
        state.current_time = float(api.get_current_time())
        volume_error = float(
            np.max(
                abs(
                    sim.porosity
                    - phi0
                    - (volumes @ (initial_m - mineral)) * sim.porosity_update_mask
                )
            )
        )
        diagnostics["max_volume_error"] = max(diagnostics["max_volume_error"], volume_error)
        if should_save_time_step(sim, state.logical_step):
            speciate(endpoint, sim.porosity, np.ones(sim.nxyz, dtype=bool))
            if np.max(abs(sim.selected_output[mineral_rows] - mineral)) > 1e-10:
                raise CouplingError(
                    "Native PHREEQC mineral state differs from accepted implicit inventory"
                )
        sim.selected_output[mineral_rows] = mineral
        sim.selected_output[delta_rows] = mineral - old_m
        save_time_step_results(
            sim, state.logical_step, state.current_time, current_k11=state.current_k11
        )
        state.logical_step += 1

    try:
        advance_to_end(sim, state, step)
    finally:
        if cell_kernel is not None:
            cell_kernel.close()
    rm.SetSaturation(sim.saturation)
    rm.SetConcentrations(np.array(native.ptrs).ravel())
    finalize_results(sim, state.logical_step, start)
    diagnostics.update(
        mean_newton_iterations=float(np.mean(iterations)), max_newton_iterations=max(iterations)
    )


class SpeciationTangent:
    """Fixed-activity-coefficient tangent; full PHREEQC residuals verify every step.

    For species composition B and concentrations s, dC = B diag(s) B.T dmu.
    Use an SVD of B sqrt(s), avoiding the squared conditioning of normal equations.
    This resolves trace redox pools without finite differences of large H/O totals.
    """

    def __init__(self, rm, components, mineral_stoichiometry):
        rm.SetSpeciesSaveOn(True)
        rm.FindComponents()
        self.rm = rm
        self.names = rm.GetSpeciesNames()
        composition = rm.GetSpeciesStoichiometry()
        self.b = np.array([[composition[s].get(c, 0) for s in self.names] for c in components])
        self.nu = np.asarray(mineral_stoichiometry)

    def derivative(self, directions, ncell, cells=None):
        species = np.asarray(self.rm.GetSpeciesConcentrations()).reshape(len(self.names), ncell)
        if cells is not None:
            species = species[:, cells]
        weighted = self.b[None, :, :] * np.sqrt(np.maximum(species.T[:, None, :], 0))
        u, singular, _ = np.linalg.svd(weighted, full_matrices=False)
        threshold = singular[:, :1] * 1e-12
        inverse = np.divide(1.0, singular, out=np.zeros_like(singular), where=singular > threshold)
        left = np.einsum("ir,nik->nrk", self.nu, u) * inverse[:, None, :]
        directions = np.array([d for _, _, d in directions]).T
        right = np.einsum("id,nik->ndk", directions, u) * inverse[:, None, :]
        return np.einsum("nrk,ndk->rdn", left, right)


class CellSpeciation:
    """Reusable one-cell PHREEQC kernel for block solves.

    The public solver permits only aqueous speciation and externally integrated
    kinetics here. No immobile equilibria are copied or silently omitted.
    """

    def __init__(self, sim, nu, rows):
        from mf6pqc.backends import CheckedPhreeqcRM

        self.sim, self.rows = sim, rows
        rm = self.rm = CheckedPhreeqcRM(sim.backend_factory.create_phreeqcrm(1, 1))
        rm.SetUnitsSolution(2)
        rm.SetComponentH2O(False)
        rm.UseSolutionDensityVolume(False)
        rm.SetRebalanceFraction(0)
        rm.SetScreenOn(False)
        rm.SetPrintChemistryOn(False, False, False)
        rm.SetSelectedOutputOn(True)
        rm.SetPorosity(sim.porosity[:1])
        rm.SetSaturation(np.ones(1))
        rm.SetDensityUser(np.ones(1))
        rm.LoadDatabase(str(sim.db_path))
        rm.RunFile(True, True, True, str(sim.pqi_path))
        rm.RunString(True, False, False, "DELETE; -all\nEND\n")
        # Create our own aqueous template. Do not assume the user's input
        # happens to contain SOLUTION 0 (or a specific initial-condition map).
        template = ["SOLUTION 0", "units mol/L", "pH 7"]
        template += [f"{name} 1e-20" for name in sim.components if name not in {"H", "O", "Charge"}]
        rm.RunString(False, True, False, "\n".join(template) + "\nEND\n")
        rm.FindComponents()
        if list(rm.GetComponents()) != sim.components:
            raise CouplingError("Cell speciation component order differs from the main backend")
        rm.InitialPhreeqc2Module(np.array([0, -1, -1, -1, -1, -1, -1], dtype=np.int32))
        rm.SetTimeStep(0)
        self.tangent = SpeciationTangent(rm, sim.components, nu)
        self.last = None

    def evaluate(self, c, cell, phi, time_days):
        key = (cell, float(phi), float(time_days), c.tobytes())
        if key != self.last:
            rm, sim = self.rm, self.sim
            rm.SetPorosity(np.array([phi]))
            rm.SetTemperature(sim.temperature[cell : cell + 1])
            rm.SetPressure(sim.pressure[cell : cell + 1])
            rm.SetDensityUser(sim.density[cell : cell + 1])
            rm.SetTime(time_days * 86400)
            rm.SetConcentrations(c.ravel())
            rm.RunCells()
            sim.implicit_diagnostics["chemistry_calls"] += 1
            back = np.asarray(rm.GetConcentrations())
            drift = float(np.max(abs(back - c.ravel())))
            sim.implicit_diagnostics["max_speciation_drift"] = max(
                sim.implicit_diagnostics["max_speciation_drift"], drift
            )
            if drift > sim.config.implicit.concentration_tolerance:
                raise CouplingError("Cell speciation changed a transported component inventory")
            self.values = np.asarray(rm.GetSelectedOutput())[self.rows, None] * np.log(10)
            self.last = key
        return self.values.copy()

    def derivative(self, directions):
        return self.tangent.derivative(directions, 1)

    def close(self):
        self.rm.MpiWorkerBreak()


class TransportResponse:
    """Apply -A_i**-1 V without assuming flow direction or equal components.

    Identical operators share a sparse factorization. Small problems retain a
    dense response; large problems use matrix products without forming an inverse.
    """

    def __init__(self, matrices, volume, stoichiometry, dense_limit, *, source_masks=None):
        from scipy.sparse.linalg import splu

        self.nu = stoichiometry
        self.ncomp, self.nreaction = self.nu.shape
        self.n = len(volume)
        self.volume = volume
        self.dense = self.n * self.nreaction <= dense_limit
        masks = (
            np.ones((self.ncomp, self.n), dtype=bool)
            if source_masks is None
            else np.asarray(source_masks, dtype=bool)
        )
        if masks.shape != (self.ncomp, self.n):
            raise ValueError("source_masks must have shape (ncomp, ncell)")
        self.groups, factors, self.group_volumes = [], [], []
        for i, matrix in enumerate(matrices):
            if not np.any(self.nu[i]):
                continue
            matrix = matrix.tocsr()
            for g, (other, _) in enumerate(self.groups):
                if (
                    np.array_equal(matrix.indptr, other.indptr)
                    and np.array_equal(matrix.indices, other.indices)
                    and np.array_equal(matrix.data, other.data)
                    and np.array_equal(volume * masks[i], self.group_volumes[g])
                ):
                    self.groups[g][1].append(i)
                    break
            else:
                self.groups.append((matrix, [i]))
                factors.append(splu(matrix.tocsc()))
                self.group_volumes.append(volume * masks[i])
        self.factors = factors
        self.responses = (
            [f.solve(-np.diag(v)) for f, v in zip(factors, self.group_volumes, strict=True)]
            if self.dense
            else None
        )
        self.directions = []
        for r in range(self.nreaction):
            for g, (_, indices) in enumerate(self.groups):
                direction = np.zeros(self.ncomp)
                direction[indices] = self.nu[indices, r]
                if np.any(direction):
                    self.directions.append((r, g, direction))

        self.group_directions = [
            (
                [d for d, (_, group, _) in enumerate(self.directions) if group == g],
                [r for r, group, _ in self.directions if group == g],
            )
            for g in range(len(self.groups))
        ]
        self.direction_matrix = np.array([d for _, _, d in self.directions]).T.reshape(
            self.ncomp, -1
        )

    def local(self, cell):
        """Diagonal transport response without new sparse factorizations."""
        result = object.__new__(TransportResponse)
        result.nu = self.nu
        result.ncomp, result.nreaction, result.n = self.ncomp, self.nreaction, 1
        result.volume = self.volume[cell : cell + 1]
        result.dense = True
        result.directions = self.directions
        result.direction_matrix = self.direction_matrix
        result.group_directions = self.group_directions
        result.groups = [
            (matrix[cell : cell + 1, cell : cell + 1], indices) for matrix, indices in self.groups
        ]
        result.group_volumes = [v[cell : cell + 1] for v in self.group_volumes]
        result.responses = [
            np.array([[-v[cell] / matrix[cell, cell]]])
            for v, (matrix, _) in zip(self.group_volumes, self.groups, strict=True)
        ]
        return result

    def apply(self, rates):
        z = np.empty((len(self.directions), self.n))
        for g, (indices, reactions) in enumerate(self.group_directions):
            rhs = rates[reactions].T
            z[indices] = (
                self.responses[g] @ rhs
                if self.dense
                else self.factors[g].solve(-self.group_volumes[g][:, None] * rhs)
            ).T
        return self.direction_matrix @ z, z

    def newton_step(self, residual, u, powers, dt, a, derivative, direct, exhausted, *, dm_du=None):
        from scipy.linalg import solve, solve_triangular
        from scipy.sparse.linalg import LinearOperator, gmres

        nr, n = residual.shape
        dm = powers * u ** (powers - 1) if dm_du is None else dm_du

        def product(flat):
            du = flat.reshape(nr, n)
            _, dz = self.apply(-dm * du / dt)
            dd = np.einsum("rdn,dn->rn", derivative, dz)
            if direct is not None:
                dd += np.einsum("rsn,sn->rn", direct, dm * du)
            result = du + dt * a * dd
            result[exhausted] = du[exhausted]
            return result.ravel()

        if self.dense:
            jac = np.eye(nr * n).reshape(nr, n, nr, n)
            for d, (s, g, _) in enumerate(self.directions):
                for r in range(nr):
                    jac[r, :, s, :] -= (
                        (a[r] * derivative[r, d])[:, None] * self.responses[g] * dm[s][None, :]
                    )
            if direct is not None:
                diagonal = np.arange(n)
                for r in range(nr):
                    for s in range(nr):
                        jac[r, diagonal, s, diagonal] += dt * a[r] * direct[r, s] * dm[s]
            jac = jac.reshape(nr * n, nr * n)
            flat_exhausted = exhausted.ravel()
            jac[flat_exhausted] = np.eye(nr * n)[flat_exhausted]
            if not np.any(np.triu(jac, 1)):
                delta = solve_triangular(jac, -residual.ravel(), lower=True, check_finite=False)
            else:
                delta = solve(jac, -residual.ravel(), check_finite=False)
        else:
            operator = LinearOperator((nr * n, nr * n), matvec=product)
            delta, info = gmres(
                operator,
                -residual.ravel(),
                rtol=1e-8,
                atol=1e-12,
                restart=min(100, nr * n),
                maxiter=100,
            )
            if info:
                raise ConvergenceError(f"Implicit Newton linear solve failed (GMRES {info})")
        if not np.all(np.isfinite(delta)):
            raise ConvergenceError("Implicit Newton produced a non-finite update")
        return delta.reshape(nr, n)


def solve_reactions(
    base,
    old_m,
    previous_rate,
    dt,
    a,
    exponent,
    transport,
    evaluate,
    nonnegative,
    options,
    *,
    cached_derivative=None,
    custom=False,
    minimum_amounts=None,
):
    """Transformed implicit kinetics with mineral exhaustion and line search.

    evaluate(c, m) returns cell-local driving forces. Only an accepted Newton
    solution is returned. Failure never commits a partially converged step.
    """
    powers = 1 / (1 - exponent)
    minimum_amounts = (
        np.zeros_like(old_m)
        if minimum_amounts is None
        else np.broadcast_to(minimum_amounts, old_m.shape)
    )
    minimum_u = minimum_amounts ** (1 - exponent)
    old_u = old_m ** (1 - exponent)
    u = np.maximum(minimum_amounts, np.maximum(old_m - previous_rate * dt, old_m * 0.1)) ** (
        1 - exponent
    )
    inactive = (old_m == 0) & (exponent > 0)
    scale = options.absolute_tolerance + options.relative_tolerance * abs(old_u)
    derivative, direct = cached_derivative, None
    if custom:
        derivative = None
    last_z = last_drive = None

    def candidate(v):
        m = v**powers
        increment, z = transport.apply((old_m - m) / dt)
        c = base + increment
        if not np.all(np.isfinite(c)) or np.any(c[nonnegative] < 0):
            return None
        drive = evaluate(c, m)
        target = old_u - dt * a * drive
        exhausted = (target <= minimum_u) | inactive
        f = v - np.where(exhausted, minimum_u, target)
        return m, c, z, drive, f, exhausted

    value = candidate(u)
    if value is None:
        u = old_u.copy()
        value = candidate(u)
    if value is None:
        raise ConvergenceError(
            "Unreacted transport has negative concentrations; check GWT discretization"
        )
    for iteration in range(options.maximum_iterations):
        m, c, z, drive, residual, exhausted = value
        error = float(np.max(abs(residual) / scale))
        if error <= 1:
            return m, c, derivative, iteration + 1, float(np.max(abs(residual)))
        refresh = (
            derivative is None
            or iteration % options.derivative_refresh == options.derivative_refresh - 1
        )
        if refresh:
            derivative = np.empty((len(old_m), len(transport.directions), base.shape[1]))
            h = options.derivative_step
            for d, (_, _, direction) in enumerate(transport.directions):
                positive = np.full(base.shape[1], h)
                negative = positive.copy()
                for i in np.flatnonzero(nonnegative & (direction < 0)):
                    positive = np.minimum(positive, 0.5 * c[i] / -direction[i])
                for i in np.flatnonzero(nonnegative & (direction > 0)):
                    negative = np.minimum(negative, 0.5 * c[i] / direction[i])
                step = np.where(positive >= negative, positive, -negative)
                if np.any(abs(step) <= 1e-20):
                    raise ConvergenceError(
                        "Cannot differentiate chemistry at a depleted component; reduce the time step"
                    )
                derivative[:, d] = (evaluate(c + direction[:, None] * step, m) - drive) / step
            if custom:
                direct = np.empty((len(old_m), len(old_m), base.shape[1]))
                for s in range(len(old_m)):
                    perturbed = m.copy()
                    step = np.maximum(h, abs(m[s]) * 1e-7)
                    perturbed[s] += step
                    direct[:, s] = (evaluate(c, perturbed) - drive) / step
        elif last_z is not None and not custom:
            dz, dd = z - last_z, drive - last_drive
            norm = np.sum(dz * dz, axis=0)
            correction = dd - np.einsum("rdn,dn->rn", derivative, dz)
            if len(transport.directions) == len(old_m) == 1:
                secant = np.divide(
                    dd[0], dz[0], out=derivative[0, 0].copy(), where=abs(dz[0]) > 1e-12
                )
                valid = (
                    (secant * derivative[0, 0] > 0)
                    & (abs(secant) > abs(derivative[0, 0]) * 0.2)
                    & (abs(secant) < abs(derivative[0, 0]) * 5)
                )
                derivative[0, 0] = np.where(valid, secant, derivative[0, 0])
            else:
                factor = np.divide(
                    correction, norm, out=np.zeros_like(correction), where=norm > 1e-24
                )
                derivative += factor[:, None, :] * dz[None, :, :]
        delta = transport.newton_step(residual, u, powers, dt, a, derivative, direct, exhausted)
        last_z, last_drive = z.copy(), drive.copy()
        alpha = 1.0
        for _ in range(24):
            trial_u = np.maximum(minimum_u, u + alpha * delta)
            trial = candidate(trial_u)
            if trial is not None:
                trial_error = float(np.max(abs(trial[4]) / scale))
                if trial_error < error or trial_error <= 1:
                    u, value = trial_u, trial
                    break
            alpha *= 0.5
        else:
            if refresh:
                raise ConvergenceError(
                    f"Implicit Newton line search failed (scaled residual {error:g})"
                )
            derivative = None
            last_z = last_drive = None
    raise ConvergenceError(
        f"Implicit reactions did not converge after {options.maximum_iterations} iterations (scaled residual {error:g})"
    )


def _solve_log_reactions(
    base,
    old_m,
    previous_rate,
    dt,
    a,
    exponent,
    transport,
    evaluate,
    nonnegative,
    options,
    *,
    cached_derivative=None,
    custom=False,
    maximum_backtracks=30,
    minimum_amounts=None,
):
    """Logarithmic kinetic residual with increments of m**(1-p) as unknowns.

    The concentration response stays linear in mineral amounts. Using log(SR)
    avoids a vanishing reaction derivative on the reduced side of a redox front.
    Every accepted step satisfies the original transformed backward Euler law.
    """
    powers = 1 / (1 - exponent)
    minimum_amounts = (
        np.zeros_like(old_m)
        if minimum_amounts is None
        else np.broadcast_to(minimum_amounts, old_m.shape)
    )
    minimum_u = minimum_amounts ** (1 - exponent)
    old_u = old_m ** (1 - exponent)
    b = dt * a
    inactive = (a == 0) | ((old_m == 0) & (exponent > 0))
    safe_b = np.where(inactive, 1, b)
    scale = options.absolute_tolerance + options.relative_tolerance * abs(old_u)
    log_min = np.minimum(-8.0, np.log(1e-3 * scale / np.maximum(b, 1e-300)))
    floor = np.maximum(np.log(np.maximum(1 - (old_u - minimum_u) / safe_b, 1e-100)), log_min)
    lower = b * np.expm1(floor)
    lower[inactive] = 0
    w = np.maximum(np.maximum(old_m - previous_rate * dt, 0) ** (1 - exponent) - old_u, lower)
    w[inactive] = 0
    derivative = cached_derivative
    augmented_attempted = False

    def affinity(c, m):
        return np.maximum(evaluate(c, m), log_min)

    def candidate(v):
        u = np.maximum(old_u + v, 0)
        m = u**powers
        increment, directions = transport.apply((old_m - m) / dt)
        c = base + increment
        if not np.all(np.isfinite(c)) or np.any(c[nonnegative] < 0):
            return None
        log_sr = affinity(c, m)
        denominator = np.maximum(b + v, safe_b * np.exp(floor))
        log_target = np.log(denominator / safe_b)
        exhausted = (log_sr <= floor) | inactive
        residual = denominator * (log_target - log_sr)
        residual[exhausted] = (v - lower)[exhausted]
        physical = v - np.maximum(minimum_u - old_u, b * np.expm1(np.minimum(log_sr, 700)))
        physical[inactive] = 0
        return u, m, c, log_sr, residual, exhausted, physical, denominator, directions

    value = candidate(w)
    # Preserve information from the preceding step. A single depleted aqueous
    # component must not discard a useful global predictor for every reaction.
    # The segment back to the old mineral state contains a feasible endpoint.
    for _ in range(30):
        if value is not None:
            break
        w *= 0.5
        value = candidate(w)
    if value is None:
        w = np.zeros_like(old_m)
        value = candidate(w)
    if value is None:
        raise ConvergenceError("Unreacted transport has negative concentrations")
    for iteration in range(options.maximum_iterations):
        u, m, c, log_sr, residual, exhausted, physical, denominator, directions = value
        if np.max(abs(physical) / scale) <= 1:
            return m, c, derivative, iteration + 1, float(np.max(abs(physical)))
        if (
            not augmented_attempted
            and transport.n > 1
            and iteration >= 4
            and np.max(abs(physical) / scale) < 1e3
            and hasattr(evaluate, "component_jacobian")
        ):
            augmented_attempted = True
            refined = solve_augmented(
                (
                    base,
                    old_m,
                    previous_rate,
                    dt,
                    a,
                    exponent,
                    transport,
                    evaluate,
                    nonnegative,
                    options,
                ),
                {"minimum_amounts": minimum_amounts},
                mineral_guess=m,
                concentrations_guess=c,
            )
            if refined is not None:
                return (*refined[:3], iteration + refined[3], refined[4])
        merit = np.linalg.norm((physical / scale).ravel())
        refresh = (
            derivative is None
            or iteration % options.derivative_refresh == options.derivative_refresh - 1
        )
        approximate = hasattr(evaluate, "jacobian")
        polish = approximate and np.max(abs(physical) / scale) < 1e3
        if refresh and approximate and not polish:
            derivative = evaluate.jacobian(c, m)
            derivative = np.where((log_sr > log_min)[:, None, :], derivative, 0)
        elif refresh:
            derivative = (
                evaluate.jacobian(c, m)
                if approximate
                else np.empty((len(old_m), len(transport.directions), base.shape[1]))
            )
            for d, (_, _, direction) in enumerate(transport.directions):
                positive = np.full(base.shape[1], options.derivative_step)
                negative = positive.copy()
                for i in np.flatnonzero(nonnegative & (direction < 0)):
                    positive = np.minimum(positive, 0.01 * c[i] / -direction[i])
                for i in np.flatnonzero(nonnegative & (direction > 0)):
                    negative = np.minimum(negative, 0.01 * c[i] / direction[i])
                h = np.minimum(positive, negative)
                valid = abs(h) > 1e-20
                if not np.all(valid) and not approximate:
                    raise ConvergenceError("Cannot differentiate depleted aqueous component")
                h = np.where(valid, h, 0.0)
                measured = (affinity(c + direction[:, None] * h, m) - log_sr) / np.where(
                    valid, h, 1.0
                )
                derivative[:, d] = np.where(valid, measured, derivative[:, d])
        dm = powers * u ** (powers - 1)
        dm[inactive] = 0
        sr = np.exp(np.minimum(log_sr, 700))
        kinetic_exhausted = (old_u - b + b * sr <= minimum_u) | inactive
        deltas = [
            transport.newton_step(
                physical, u, powers, dt, -a * sr, derivative, None, kinetic_exhausted, dm_du=dm
            )
        ]
        accepted = False
        for mode in range(2):
            if mode:
                deltas.append(
                    transport.newton_step(
                        residual,
                        u,
                        powers,
                        dt,
                        -denominator / dt,
                        derivative,
                        None,
                        exhausted,
                        dm_du=dm,
                    )
                )
            delta = deltas[-1]
            # Projection alone can trap Newton at an inventory lower bound.
            # Recompute the free-variable direction with those bounds active.
            blocked = np.zeros_like(w, dtype=bool)
            for _ in range(len(old_m) * 2):
                new_blocked = (delta < lower - w - 1e-14) & (
                    w - lower < np.maximum(1e-12, scale * 1e-3)
                )
                if not np.any(new_blocked & ~blocked):
                    break
                blocked |= new_blocked
                constrained = (residual if mode else physical).copy()
                constrained[blocked] = (w - lower)[blocked]
                delta = transport.newton_step(
                    constrained,
                    u,
                    powers,
                    dt,
                    -denominator / dt if mode else -a * sr,
                    derivative,
                    None,
                    (exhausted if mode else kinetic_exhausted) | blocked,
                    dm_du=dm,
                )
            alpha = 1.0
            for _ in range(maximum_backtracks):
                trial_w = np.maximum(lower, w + alpha * delta)
                trial_w[inactive] = 0
                trial = candidate(trial_w)
                if trial is not None and (
                    np.linalg.norm((trial[6] / scale).ravel()) < merit * (1 - 1e-4 * alpha)
                    or np.max(abs(trial[6]) / scale) <= 1
                ):
                    w, value = trial_w, trial
                    accepted = True
                    break
                alpha *= 0.5
            if accepted:
                break
        if not accepted:
            if not refresh:
                derivative = None
                continue
            error = ConvergenceError(
                f"Logarithmic implicit line search failed (molar residual {np.max(abs(physical)):g})"
            )
            error.mineral_guess = m.copy()
            error.concentrations_guess = c.copy()
            raise error
    error = ConvergenceError(
        f"Logarithmic implicit reactions did not converge (molar residual {np.max(abs(physical)):g})"
    )
    error.mineral_guess = m.copy()
    error.concentrations_guess = c.copy()
    raise error


def _block_start(args, kwargs):
    """Nonlinear block sweeps using the complete transport inverse.

    All minerals in one cell are solved together. Updating that cell changes
    concentrations throughout the domain via the original transport response.
    A global Newton check is required before any result can be returned.
    """
    from dataclasses import replace

    base, old, previous_rate, dt, a, exponent, transport, evaluate, nonnegative, options = args
    if not transport.dense or transport.n == 1 or not hasattr(evaluate, "at_cells"):
        return None
    minimum = kwargs.get("minimum_amounts")
    minimum = np.zeros_like(old) if minimum is None else np.broadcast_to(minimum, old.shape)
    guess, c = old.copy(), base.copy()
    for sweep in range(8):
        order = range(transport.n) if sweep % 2 == 0 else range(transport.n - 1, -1, -1)
        for i in order:
            if np.all(a[:, i] == 0):
                continue
            local = transport.local(i)
            # This is (A^-1)_ii, not 1/A_ii: neighbouring-cell transport remains
            # part of the reduced block equation, including cyclic diffusion.
            local.responses = [np.array([[response[i, i]]]) for response in transport.responses]
            local_base = (
                c[:, i : i + 1] - local.apply((old[:, i : i + 1] - guess[:, i : i + 1]) / dt)[0]
            )

            def local_evaluate(lc, lm, i=i, c=c, guess=guess):
                trial_c, trial_m = c.copy(), guess.copy()
                trial_c[:, i], trial_m[:, i] = lc[:, 0], lm[:, 0]
                return evaluate.at_cells(trial_c, trial_m, [i])

            if hasattr(evaluate, "jacobian_at_cells"):

                def local_jacobian(lc, lm, i=i, c=c, guess=guess, local=local):
                    trial_c, trial_m = c.copy(), guess.copy()
                    trial_c[:, i], trial_m[:, i] = lc[:, 0], lm[:, 0]
                    return evaluate.jacobian_at_cells(trial_c, trial_m, [i], local.directions)

                local_evaluate.jacobian = local_jacobian
            try:
                result = solve_log_reactions(
                    local_base,
                    old[:, i : i + 1],
                    (old[:, i : i + 1] - guess[:, i : i + 1]) / dt,
                    dt,
                    a[:, i : i + 1],
                    exponent,
                    local,
                    local_evaluate,
                    nonnegative,
                    options,
                    minimum_amounts=minimum[:, i : i + 1],
                )
            except ConvergenceError:
                continue
            change = np.zeros_like(old)
            change[:, i] = result[0][:, 0] - guess[:, i]
            dc = transport.apply(-change / dt)[0]
            decreasing = (dc < 0) & nonnegative[:, None]
            fraction = max(
                0.0, min(1.0, np.min(-c[decreasing] / dc[decreasing], initial=np.inf) * 0.999999)
            )
            guess += fraction * change
            c += fraction * dc
        c = base + transport.apply((old - guess) / dt)[0]
        if np.any(c[nonnegative] < 0):
            guess = (guess + old) * 0.5
            c = base + transport.apply((old - guess) / dt)[0]
        evaluate.block_sweeps = getattr(evaluate, "block_sweeps", 0) + 1
        try:
            return _solve_log_reactions(
                base,
                old,
                (old - guess) / dt,
                dt,
                a,
                exponent,
                transport,
                evaluate,
                nonnegative,
                replace(options, maximum_iterations=min(30, options.maximum_iterations)),
                **dict(kwargs, cached_derivative=None),
            )
        except ConvergenceError as error:
            guess = getattr(error, "mineral_guess", guess)
            c = base + transport.apply((old - guess) / dt)[0]
    return None


def _coordinate_start(args, kwargs):
    """Bracket individual reactions while retaining the full transport response.

    This is only a starting-point strategy. A full Newton solve must subsequently
    satisfy the declared tolerance; scalar roots are not accepted as a time step.
    """
    from dataclasses import replace

    from scipy.optimize import brentq

    base, old, previous_rate, dt, a, exponent, transport, evaluate, nonnegative, options = args
    if not transport.dense or transport.n == 1 or not hasattr(evaluate, "at_cells"):
        return None
    alpha = 1 - exponent
    old_u, b = old**alpha, dt * a
    minimum = kwargs.get("minimum_amounts")
    minimum = np.zeros_like(old) if minimum is None else np.broadcast_to(minimum, old.shape)
    minimum_u = minimum**alpha
    guess, c = old.copy(), base.copy()
    for sweep in range(16):
        order = range(transport.n) if sweep % 2 == 0 else range(transport.n - 1, -1, -1)
        for i in order:
            for r in range(len(old)):
                if b[r, i] == 0 or (old[r, i] == 0 and exponent[r, 0] > 0):
                    continue
                current = guess[r, i]
                influence = np.zeros_like(base)
                for reaction, g, direction in transport.directions:
                    if reaction == r:
                        influence += direction[:, None] * transport.responses[g][None, :, i] / dt
                if not np.any(influence):
                    continue

                def scalar(m, c=c, i=i, r=r, influence=influence, current=current, guess=guess):
                    trial_c = c.copy()
                    trial_c[:, i] -= influence[:, i] * (m - current)
                    log_sr = evaluate.at_cells(trial_c, guess, [i])[r, 0]
                    return m ** alpha[r, 0] - max(
                        minimum_u[r, i], old_u[r, i] + b[r, i] * np.expm1(min(log_sr, 100))
                    )

                if abs(scalar(current)) < options.absolute_tolerance * 0.2:
                    continue
                low = max(minimum_u[r, i], old_u[r, i] - b[r, i]) ** (1 / alpha[r, 0])
                high = np.inf
                for component in np.flatnonzero(nonnegative):
                    v = influence[component]
                    values = c[component]
                    if np.any(v > 0):
                        high = min(high, current + np.min(values[v > 0] / v[v > 0]))
                    if np.any(v < 0):
                        low = max(low, current + np.max(values[v < 0] / v[v < 0]))
                if not np.isfinite(high):
                    continue
                high = max(current, high - (high - current) * 1e-9)
                low = min(current, low)
                left, right = scalar(low), scalar(high)
                if left >= 0:
                    root = low
                elif right <= 0:
                    root = high
                else:
                    root = brentq(scalar, low, high, xtol=1e-16, rtol=1e-14, maxiter=200)
                guess[r, i] = root
                c -= influence * (root - current)
        # Remove accumulated floating-point drift before the global solve.
        c = base + transport.apply((old - guess) / dt)[0]
        if np.any(c[nonnegative] < 0):
            guess = (guess + old) * 0.5
            c = base + transport.apply((old - guess) / dt)[0]
        evaluate.coordinate_sweeps = getattr(evaluate, "coordinate_sweeps", 0) + 1
        try:
            return _solve_log_reactions(
                base,
                old,
                (old - guess) / dt,
                dt,
                a,
                exponent,
                transport,
                evaluate,
                nonnegative,
                replace(options, maximum_iterations=min(30, options.maximum_iterations)),
                **dict(kwargs, cached_derivative=None),
            )
        except ConvergenceError as error:
            guess = getattr(error, "mineral_guess", guess)
            c = base + transport.apply((old - guess) / dt)[0]
    return None


def solve_log_reactions(*args, **kwargs):
    """Continue reaction strength if the full nonlinear step needs a safer start."""
    try:
        return _solve_log_reactions(*args, **kwargs)
    except ConvergenceError as error:
        # A stalled one-cell solve may need independent trace concentrations
        # to avoid cancellation in the mineral-to-aqueous transport response.
        if args[6].n == 1 and hasattr(error, "mineral_guess"):
            refined = solve_augmented(
                args,
                kwargs,
                mineral_guess=error.mineral_guess,
                concentrations_guess=error.concentrations_guess,
            )
            if refined is not None:
                return refined
        predicted = _block_start(args, kwargs)
        if predicted is None:
            predicted = _coordinate_start(args, kwargs)
        if predicted is not None:
            return predicted
        old_m, dt = args[1], args[3]
        guess = old_m.copy()
        factor, increment, iterations = 0.0, 0.001, 0
        for _ in range(100):
            target = min(1.0, factor + increment)
            trial = list(args)
            trial[2] = (old_m - guess) / dt
            trial[4] = args[4] * target
            try:
                result = _solve_log_reactions(*trial, **dict(kwargs, cached_derivative=None))
            except ConvergenceError:
                increment *= 0.5
                if increment < 1e-10:
                    raise
            else:
                guess = result[0]
                factor = target
                iterations += result[3]
                if factor == 1:
                    return (*result[:3], iterations, result[4])
                increment = min(increment * 2, 0.25)
        raise ConvergenceError(
            "Implicit reaction-strength continuation did not reach the full step"
        ) from None


def solve_directed_reactions(
    base,
    old_m,
    previous_rate,
    dt,
    a,
    exponent,
    transport,
    evaluate,
    nonnegative,
    options,
    *,
    minimum_amounts=None,
):
    """Block substitution for acyclic transport graphs; None for cyclic graphs.

    Ordering is inferred from the union of component-matrix dependencies, so
    reversed flow, branched grids, and arbitrary cell numbering are supported.
    """
    import heapq

    n = transport.n
    dependencies = [set() for _ in range(n)]
    followers = [set() for _ in range(n)]
    for matrix, _ in transport.groups:
        for i in range(n):
            for j, v in zip(
                matrix.indices[matrix.indptr[i] : matrix.indptr[i + 1]],
                matrix.data[matrix.indptr[i] : matrix.indptr[i + 1]],
                strict=True,
            ):
                if i != j and v != 0:
                    dependencies[i].add(int(j))
                    followers[j].add(i)
    if not any(dependencies):
        return None
    degree = np.array([len(s) for s in dependencies])
    ready = list(np.flatnonzero(degree == 0))
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in followers[i]:
            degree[j] -= 1
            if degree[j] == 0:
                heapq.heappush(ready, j)
    if len(order) != n:
        return None
    from dataclasses import replace

    try:
        return _solve_log_reactions(
            base,
            old_m,
            previous_rate,
            dt,
            a,
            exponent,
            transport,
            evaluate,
            nonnegative,
            replace(options, maximum_iterations=min(2, options.maximum_iterations)),
            maximum_backtracks=4,
            minimum_amounts=minimum_amounts,
        )
    except ConvergenceError as error:
        if hasattr(error, "mineral_guess"):
            previous_rate = (old_m - error.mineral_guess) / dt
    evaluate.directed_blocks_used = True
    c = base.copy()
    mineral = old_m.copy()
    maximum_iterations = 0
    for i in order:
        local_base = base[:, i : i + 1].copy()
        for matrix, indices in transport.groups:
            row = matrix.getrow(i)
            diagonal = matrix[i, i]
            for component in indices:
                local_base[component, 0] -= (
                    row @ (c[component] - base[component])
                ).item() / diagonal
        if np.all(a[:, i] == 0):
            c[:, i] = local_base[:, 0]
            continue
        local_transport = transport.local(i)

        def local_evaluate(lc, lm, i=i):
            trial_c, trial_m = c.copy(), mineral.copy()
            trial_c[:, i] = lc[:, 0]
            trial_m[:, i] = lm[:, 0]
            return evaluate.at_cells(trial_c, trial_m, [i])

        if hasattr(evaluate, "jacobian_at_cells"):

            def local_jacobian(lc, lm, directions=local_transport.directions, i=i):
                trial_c, trial_m = c.copy(), mineral.copy()
                trial_c[:, i] = lc[:, 0]
                trial_m[:, i] = lm[:, 0]
                return evaluate.jacobian_at_cells(trial_c, trial_m, [i], directions)

            local_evaluate.jacobian = local_jacobian
            local_evaluate.component_jacobian = local_jacobian
        result = solve_log_reactions(
            local_base,
            old_m[:, i : i + 1],
            previous_rate[:, i : i + 1],
            dt,
            a[:, i : i + 1],
            exponent,
            local_transport,
            local_evaluate,
            nonnegative,
            options,
            minimum_amounts=None if minimum_amounts is None else minimum_amounts[:, i : i + 1],
        )
        mineral[:, i : i + 1], c[:, i : i + 1] = result[:2]
        maximum_iterations = max(maximum_iterations, result[3])
        evaluate.augmented_iterations = getattr(evaluate, "augmented_iterations", 0) + getattr(
            local_evaluate, "augmented_iterations", 0
        )
    # Local kernels can differ slightly from the complete chemistry state.
    # Check the original kinetic equation before accepting their solution.
    old_u = old_m ** (1 - exponent)
    minimum_u = 0 if minimum_amounts is None else minimum_amounts ** (1 - exponent)
    log_sr = evaluate(c, mineral)
    target = np.maximum(minimum_u, old_u + dt * a * np.expm1(np.minimum(log_sr, 700)))
    target[(old_m == 0) & (exponent > 0)] = 0
    residual = mineral ** (1 - exponent) - target
    scale = options.absolute_tolerance + options.relative_tolerance * abs(old_u)
    if np.max(abs(residual) / scale) <= 1:
        return mineral, c, None, maximum_iterations, float(np.max(abs(residual)))
    # Refine a failed check with the original transport response and full
    # PHREEQC residual, using the block solution as the initial guess.
    result = solve_log_reactions(
        base,
        old_m,
        (old_m - mineral) / dt,
        dt,
        a,
        exponent,
        transport,
        evaluate,
        nonnegative,
        options,
        minimum_amounts=minimum_amounts,
    )
    return (*result[:3], maximum_iterations + result[3], result[4])


def solve_augmented(args, kwargs, mineral_guess=None, concentrations_guess=None):
    from scipy.sparse import bmat, diags, eye
    from scipy.sparse.linalg import spsolve

    base, old, previous_rate, dt, a, exponent, transport, evaluate, nonnegative, options = args
    if not hasattr(evaluate, "component_jacobian"):
        return None
    components = np.flatnonzero(np.any(transport.nu != 0, axis=1))
    if not np.all(nonnegative[components]):
        return None
    nr, n = len(old), transport.n
    reconstruction = None
    if len(transport.groups) == 1:
        # Prefer low-concentration independent components. Conserved aqueous
        # combinations reconstruct abundant solvent totals without introducing
        # nearly redundant H/O unknowns into the nonlinear solve.
        selected = []
        rank = 0
        for i in sorted(components, key=lambda i: np.max(abs(base[i]))):
            trial_rank = np.linalg.matrix_rank(transport.nu[selected + [i]])
            if trial_rank > rank:
                selected.append(i)
                rank = trial_rank
        if rank == np.linalg.matrix_rank(transport.nu):
            components = np.array(selected)
            reconstruction = transport.nu @ np.linalg.pinv(transport.nu[components])
    nc = len(components)
    matrices = {i: matrix for matrix, indices in transport.groups for i in indices}
    volumes = {
        i: v
        for (_, indices), v in zip(transport.groups, transport.group_volumes, strict=True)
        for i in indices
    }
    alpha = 1 - exponent
    powers = 1 / alpha
    ou = old**alpha
    b = dt * a
    minimum = kwargs.get("minimum_amounts")
    minimum = np.zeros_like(old) if minimum is None else np.broadcast_to(minimum, old.shape)
    mu = minimum**alpha
    inactive = (b == 0) | ((old == 0) & (exponent > 0))
    safe_b = np.where(inactive, 1, b)
    kinetic_scale = options.absolute_tolerance + options.relative_tolerance * abs(ou)
    floor = np.maximum(
        np.log(np.maximum(1 - (ou - mu) / safe_b, 1e-100)),
        np.minimum(-8, np.log(1e-3 * kinetic_scale / np.maximum(b, 1e-300))),
    )
    floor[inactive] = 0
    guess = old if mineral_guess is None else mineral_guess
    w = np.maximum(guess**alpha - ou, b * np.expm1(floor))
    w[inactive] = 0
    c = base + transport.apply((old - np.maximum(ou + w, 0) ** powers) / dt)[0]
    if concentrations_guess is not None:
        c = concentrations_guess.copy()
    z = np.log(np.maximum(1 + w / safe_b, np.exp(floor)))
    z[inactive] = 0
    y = np.log(np.maximum(c[components], np.maximum(abs(base[components]) * 1e-15, 1e-100)))
    row_scale = np.array(
        [
            np.asarray(abs(matrices[i]).sum(axis=1)).ravel() * np.maximum(abs(base[i]), 1e-12)
            for i in components
        ]
    )
    component_directions = (
        np.eye(transport.ncomp)[:, components] if reconstruction is None else reconstruction
    )
    basis = [(0, 0, component_directions[:, j]) for j in range(nc)]

    def candidate(y, z):
        if np.max(abs(y)) > 690 or np.max(z) > 100:
            return None
        selected_c = np.exp(y)
        c = (
            base.copy()
            if reconstruction is None
            else base + reconstruction @ (selected_c - base[components])
        )
        c[components] = selected_c
        if np.any(c[nonnegative] < 0):
            return None
        w = b * np.expm1(z)
        u = np.maximum(ou + w, 0)
        m = u**powers
        ln_sr = evaluate(c, m)
        exhausted = (ln_sr <= floor) | inactive
        fz = z - np.maximum(ln_sr, floor)
        fz[inactive] = z[inactive]
        rate = transport.nu @ ((old - m) / dt)
        fc = (
            np.array([matrices[i] @ (c[i] - base[i]) + volumes[i] * rate[i] for i in components])
            / row_scale
        )
        physical = w - np.maximum(mu - ou, b * np.expm1(np.minimum(ln_sr, 700)))
        physical[inactive] = 0
        residual = np.r_[fc.ravel(), fz.ravel()]
        return c, m, u, ln_sr, exhausted, physical, residual

    value = candidate(y, z)
    if value is None:
        return None
    for iteration in range(options.maximum_iterations):
        c, m, u, ln_sr, exhausted, physical, residual = value
        if np.max(abs(physical) / kinetic_scale) <= 1:
            closure = c - (base + transport.apply((old - m) / dt)[0])
            if np.max(abs(closure)) <= min(options.concentration_tolerance, 1e-11):
                evaluate.augmented_iterations = (
                    getattr(evaluate, "augmented_iterations", 0) + iteration + 1
                )
                return m, c, None, iteration + 1, float(np.max(abs(physical)))
        derivative = evaluate.component_jacobian(c, m, basis)
        # Correct activity-coefficient and water-balance derivatives near a root.
        if np.max(abs(residual[nc * n :])) < 1:
            for j, i in enumerate(components):
                h = np.clip(options.derivative_step / np.maximum(c[i], 1e-300), 1e-10, 1e-5)
                perturbed = c + component_directions[:, j, None] * (c[i] * np.expm1(h))
                derivative[:, j] = (evaluate(perturbed, m) - ln_sr) / (c[i] * np.expm1(h))
        dm = powers * u ** (powers - 1) * b * np.exp(z)
        dm[inactive] = 0
        cc = [[None] * nc for _ in range(nc)]
        for j, i in enumerate(components):
            cc[j][j] = diags(1 / row_scale[j]) @ matrices[i] @ diags(c[i])
        cz = [
            [diags(-volumes[i] * transport.nu[i, r] * dm[r] / dt / row_scale[j]) for r in range(nr)]
            for j, i in enumerate(components)
        ]
        zc = [
            [
                diags(np.where(exhausted[r], 0, -derivative[r, j] * c[i]))
                for j, i in enumerate(components)
            ]
            for r in range(nr)
        ]
        jac = bmat(
            [
                [bmat(cc, format="csc"), bmat(cz, format="csc")],
                [bmat(zc, format="csc"), eye(nr * n, format="csc")],
            ],
            format="csc",
        )
        delta = spsolve(jac, -residual)
        if not np.all(np.isfinite(delta)):
            return None
        blocked = np.zeros((nr, n), bool)
        for _ in range(nr * 2):
            dz = delta[nc * n :].reshape(nr, n)
            new = (dz < floor - z - 1e-14) & (z - floor < 1e-9)
            if not np.any(new & ~blocked):
                break
            blocked |= new
            bound_rows = nc * n + np.flatnonzero(blocked)
            work = jac.tolil()
            rhs = -residual.copy()
            for row in bound_rows:
                work.rows[row] = [row]
                work.data[row] = [1.0]
                rhs[row] = (floor - z).ravel()[row - nc * n]
            delta = spsolve(work.tocsc(), rhs)
        dy = delta[: nc * n].reshape(nc, n)
        dz = delta[nc * n :].reshape(nr, n)
        fraction = min(1.0, 4 / max(np.max(abs(dy)), 1e-30))
        merit = np.linalg.norm(residual)
        for _ in range(30):
            yy = y + fraction * dy
            zz = np.maximum(floor, z + fraction * dz)
            zz[inactive] = 0
            trial = candidate(yy, zz)
            if trial is not None and np.linalg.norm(trial[-1]) < merit * (1 - 1e-4 * fraction):
                y, z, value = yy, zz, trial
                break
            fraction *= 0.5
        else:
            return None
    return None
