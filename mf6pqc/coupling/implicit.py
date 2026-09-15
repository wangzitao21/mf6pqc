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
from mf6pqc.coupling.implicit_system import (
    TransportResponse,
    solve_directed_reactions,
    solve_log_reactions,
    solve_reactions,
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
        from scipy.sparse import csr_matrix

        self.api.prepare_solve(self.flow)
        self.solve(self.flow)
        for i, sid in enumerate(self.ids):
            ibound = self.ptr(get_gwt_model_name(self.sim.components[i]) + "/IBOUND")
            free = ibound > 0
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
        matrices = [
            csr_matrix((v.copy(), ja, ia), shape=(self.n, self.n)) for v, ja, ia in self.csr
        ]
        return base, matrices

    def matrices(self):
        from scipy.sparse import csr_matrix

        return [csr_matrix((v.copy(), ja, ia), shape=(self.n, self.n)) for v, ja, ia in self.csr]

    def commit(self, old, concentrations, rates):
        for i, sid in enumerate(self.ids):
            if not np.any(rates[i]):
                continue
            ibound = self.ptr(get_gwt_model_name(self.sim.components[i]) + "/IBOUND")
            free = ibound > 0
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
        from mf6pqc.coupling.speciation_tangent import SpeciationTangent

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
            for ptr in sim.diffc_ptrs:
                ptr[:] = update_diffc(evaluation_phi, sim.d0)
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
                    from mf6pqc.coupling.speciation_tangent import CellSpeciation

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
        # Record the storage convention explicitly. The MIN3P lagged option
        # keeps concentration fixed when phi changes and therefore changes
        # aqueous inventory; this is separate from the reaction-source balance.
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
            sim.selected_output[delta_rows] = mineral - old_m
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
