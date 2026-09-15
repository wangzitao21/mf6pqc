"""Coupled concentration/mineral Newton solve for trace-component cancellation.

Concentrations remain independent positive unknowns. The original native
transport equations and original kinetic law both have to pass verification.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import bmat, diags, eye
from scipy.sparse.linalg import spsolve


def solve_augmented(args, kwargs, mineral_guess=None, concentrations_guess=None):
    base, old, previous_rate, dt, a, exponent, transport, evaluate, nonnegative, options = args
    if transport.n == 1 or not hasattr(evaluate, "component_jacobian"):
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
