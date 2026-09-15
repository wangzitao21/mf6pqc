"""Reduced Newton system; independent of native APIs and benchmark grids."""

from __future__ import annotations

import numpy as np

from mf6pqc.exceptions import ConvergenceError


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

    def local(self, cell):
        """Diagonal transport response without new sparse factorizations."""
        result = object.__new__(TransportResponse)
        result.nu = self.nu
        result.ncomp, result.nreaction, result.n = self.ncomp, self.nreaction, 1
        result.volume = self.volume[cell : cell + 1]
        result.dense = True
        result.directions = self.directions
        result.responses = [
            np.array([[-v[cell] / matrix[cell, cell]]])
            for v, (matrix, _) in zip(self.group_volumes, self.groups, strict=True)
        ]
        return result

    def response(self, group, rate):
        if self.dense:
            return self.responses[group] @ rate
        return self.factors[group].solve(-self.group_volumes[group] * rate)

    def apply(self, rates):
        z = np.array([self.response(g, rates[r]) for r, g, _ in self.directions])
        concentrations = np.zeros((self.ncomp, self.n))
        for zd, (_, _, direction) in zip(z, self.directions, strict=True):
            concentrations += direction[:, None] * zd
        return concentrations, z

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
    # Preserve information from the preceding step. A single depleted aqueous
    # component must not discard a useful global predictor for every reaction.
    # The segment back to the old mineral state contains a feasible endpoint.
    for _ in range(30):
        if value is not None:
            break
        w *= 0.5
        value = candidate(w)
    if value is None:
        raise ConvergenceError("Unreacted transport has negative concentrations")
    for iteration in range(options.maximum_iterations):
        u, m, c, log_sr, residual, exhausted, physical, denominator, directions = value
        if np.max(abs(physical) / scale) <= 1:
            return m, c, derivative, iteration + 1, float(np.max(abs(physical)))
        if (
            not augmented_attempted
            and iteration >= 4
            and np.max(abs(physical) / scale) < 1e3
            and hasattr(evaluate, "component_jacobian")
        ):
            from mf6pqc.coupling.implicit_augmented import solve_augmented

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
            raise error
    error = ConvergenceError(
        f"Logarithmic implicit reactions did not converge (molar residual {np.max(abs(physical)):g})"
    )
    error.mineral_guess = m.copy()
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
    except ConvergenceError:
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
    maximum_iterations, max_residual = 0, 0.0
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

            def local_jacobian(lc, lm, i=i, local_transport=local_transport):
                trial_c, trial_m = c.copy(), mineral.copy()
                trial_c[:, i] = lc[:, 0]
                trial_m[:, i] = lm[:, 0]
                return evaluate.jacobian_at_cells(trial_c, trial_m, [i], local_transport.directions)

            local_evaluate.jacobian = local_jacobian
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
        max_residual = max(max_residual, result[4])
    return mineral, c, None, maximum_iterations, max_residual
