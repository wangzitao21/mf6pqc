"""Independent algebraic and native checks for general implicit kinetics."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from mf6pqc import ImplicitOptions, KineticReaction
from mf6pqc.coupling.implicit import (
    TransportResponse,
    _block_start,
    _coordinate_start,
    solve_directed_reactions,
    solve_log_reactions,
    solve_reactions,
)
from mf6pqc.exceptions import ConfigurationError


class ImplicitSystemTests(unittest.TestCase):
    def test_multiple_reactions_distinct_diffusive_operators_and_cell_permutation(self):
        n, nr = 6, 2
        volume = np.linspace(0.7, 1.3, n)
        matrices = []
        for diffusion in (0.2, 0.7):
            a = np.diag(-2 - np.arange(n) / 10)
            a += np.diag(np.full(n - 1, diffusion), 1)
            a += np.diag(np.full(n - 1, diffusion / 2), -1)
            matrices.append(a)
        nu = np.array([[1.0, 0.3], [0.2, 1.0]])
        base = np.array([np.linspace(0.02, 0.05, n), np.linspace(0.1, 0.15, n)])
        old = np.full((nr, n), 10.0)
        k = np.array([[0.2], [0.1]]) * np.ones((nr, n))
        b = np.array([[2.0, 0.5], [1.0, 1.0]])
        constant = np.array([[1.0], [0.3]])
        dt = 0.7
        # Independent linear rate equation, not the Newton implementation:
        # q = k * (constant - B * (base + T nu q)).
        t = [-np.linalg.solve(a, np.diag(volume)) for a in matrices]
        lhs = np.eye(nr * n).reshape(nr, n, nr, n)
        for r in range(nr):
            for s in range(nr):
                for j in range(2):
                    lhs[r, :, s, :] += k[r, :, None] * b[r, j] * t[j] * nu[j, s]
        rhs = k * (constant - b @ base)
        exact_rate = np.linalg.solve(lhs.reshape(nr * n, nr * n), rhs.ravel()).reshape(nr, n)
        expected = old - dt * exact_rate
        for order in (np.arange(n), np.array([4, 1, 5, 0, 3, 2])):
            for dense_limit in (400, 1):
                with self.subTest(order=order, dense_limit=dense_limit):
                    response = TransportResponse(
                        [csr_matrix(a[np.ix_(order, order)]) for a in matrices],
                        volume[order],
                        nu,
                        dense_limit,
                    )
                    result, c, *_ = solve_reactions(
                        base[:, order],
                        old[:, order],
                        np.zeros_like(old),
                        dt,
                        k[:, order],
                        np.zeros((nr, 1)),
                        response,
                        lambda c, m: constant - b @ c,
                        np.ones(2, bool),
                        ImplicitOptions(absolute_tolerance=1e-10),
                    )
                    np.testing.assert_allclose(result, expected[:, order], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(
                        (old[:, order] - result) / dt,
                        k[:, order] * (constant - b @ c),
                        atol=1e-9,
                        rtol=0,
                    )

    def test_surface_transform_matches_exact_constant_driving_force_and_exhaustion(self):
        dt, p, k, m0 = 50.0, 2 / 3, 0.6, 8.0
        old = np.array([[m0, m0]])
        a = np.full_like(old, (1 - p) * k / m0**p)
        response = TransportResponse([csr_matrix(-np.eye(2))], np.ones(2), np.ones((1, 1)), 400)
        drive = np.array([[1.0, 0.1]])
        result, *_ = solve_reactions(
            np.ones((1, 2)),
            old,
            np.zeros_like(old),
            dt,
            a,
            np.full((1, 1), p),
            response,
            lambda c, m: drive,
            np.ones(1, bool),
            ImplicitOptions(),
        )
        expected = np.maximum(0, old ** (1 - p) - dt * a * drive) ** (1 / (1 - p))
        np.testing.assert_allclose(result, expected, atol=1e-10, rtol=0)
        self.assertEqual(result[0, 0], 0)

    def test_precipitation_from_zero_and_inventory_dependent_custom_rate(self):
        response = TransportResponse([csr_matrix([[-1.0]])], np.ones(1), np.ones((1, 1)), 400)
        precipitated, *_ = solve_reactions(
            np.array([[2.0]]),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            1.0,
            np.ones((1, 1)),
            np.zeros((1, 1)),
            response,
            lambda c, m: np.full((1, 1), -0.2),
            np.ones(1, bool),
            ImplicitOptions(),
        )
        np.testing.assert_allclose(precipitated, 0.2)
        decayed, *_ = solve_reactions(
            np.ones((1, 1)),
            np.array([[4.0]]),
            np.zeros((1, 1)),
            2.0,
            np.full((1, 1), 0.3),
            np.zeros((1, 1)),
            response,
            lambda c, m: m,
            np.ones(1, bool),
            ImplicitOptions(absolute_tolerance=1e-10),
            custom=True,
        )
        np.testing.assert_allclose(decayed, 4 / (1 + 2 * 0.3), atol=1e-9)

    def test_minimum_inventory_survives_dissolution_and_regrows(self):
        # An analytical constant-affinity solution tests both residual forms.
        response = TransportResponse([csr_matrix([[-1.0]])], np.ones(1), np.ones((1, 1)), 400)
        minimum = np.array([[1e-6]])
        p = np.array([[2 / 3]])
        old = np.array([[1.0]])
        a = np.array([[1.0]])
        for solver, undersaturated, supersaturated in (
            (solve_reactions, 0.9, -0.25),
            (solve_log_reactions, np.log(0.1), np.log(1.25)),
        ):
            with self.subTest(solver=solver.__name__):
                depleted = solver(
                    np.array([[5.0]]),
                    old,
                    np.zeros_like(old),
                    2.0,
                    a,
                    p,
                    response,
                    lambda c, m, value=undersaturated: np.full_like(m, value),
                    np.ones(1, bool),
                    ImplicitOptions(absolute_tolerance=1e-10),
                    minimum_amounts=minimum,
                )[0]
                np.testing.assert_allclose(depleted, minimum, atol=1e-12, rtol=0)
                regrown = solver(
                    np.array([[5.0]]),
                    depleted,
                    np.zeros_like(old),
                    0.1,
                    a,
                    p,
                    response,
                    lambda c, m, value=supersaturated: np.full_like(m, value),
                    np.ones(1, bool),
                    ImplicitOptions(absolute_tolerance=1e-10),
                    minimum_amounts=minimum,
                )[0]
                expected = (minimum ** (1 - p) + 0.1 * 0.25) ** (1 / (1 - p))
                np.testing.assert_allclose(regrown, expected, atol=1e-11, rtol=0)

    def test_logarithmic_solver_and_coordinate_start_agree_with_linear_solution(self):
        matrix = np.array([[-2.0, 0.3], [0.3, -2.0]])
        response = TransportResponse([csr_matrix(matrix)], np.ones(2), np.ones((1, 1)), 400)
        base, old, dt, a = np.array([[0.2, 0.5]]), np.ones((1, 2)), 0.7, np.ones((1, 2)) * 0.4
        # SR=2*C makes the original kinetic equation a linear, independent solve.
        t = -np.linalg.inv(matrix)
        q = np.linalg.solve(np.eye(2) + 0.8 * t, 0.4 * (1 - 2 * base[0]))
        expected = old - dt * q

        def evaluate(c, m):
            return np.log(2 * c)

        evaluate.at_cells = lambda c, m, cells: np.log(2 * c[:, cells])
        args = (
            base,
            old,
            np.zeros_like(old),
            dt,
            a,
            np.zeros((1, 1)),
            response,
            evaluate,
            np.ones(1, bool),
            ImplicitOptions(absolute_tolerance=1e-10),
        )
        for result in (
            solve_log_reactions(*args),
            _coordinate_start(args, {}),
            _block_start(args, {}),
        ):
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result[0], expected, atol=1e-9, rtol=0)
            np.testing.assert_allclose(matrix @ (result[1] - base)[0], -q, atol=1e-9)
        # Diffusion is cyclic and must never enter directed block substitution.
        self.assertIsNone(solve_directed_reactions(*args))

    def test_directed_blocks_follow_permuted_reversed_flow(self):
        n = 5
        matrix = -np.eye(n) * 2 + np.diag(np.full(n - 1, 0.5), 1)
        base = np.linspace(0.1, 0.3, n)[None, :]
        old = np.ones((1, n))
        for order in (np.arange(n), np.array([2, 0, 4, 1, 3])):
            response = TransportResponse(
                [csr_matrix(matrix[np.ix_(order, order)])], np.ones(n), np.ones((1, 1)), 400
            )

            def evaluate(c, m):
                return np.log(2 * c)

            evaluate.at_cells = lambda c, m, cells: np.log(2 * c[:, cells])
            args = (
                base[:, order],
                old,
                np.zeros_like(old),
                1.0,
                np.full_like(old, 0.3),
                np.zeros((1, 1)),
                response,
                evaluate,
                np.ones(1, bool),
                ImplicitOptions(absolute_tolerance=1e-10),
            )
            global_result = solve_log_reactions(*args)
            directed = solve_directed_reactions(*args)
            np.testing.assert_allclose(directed[0], global_result[0], atol=1e-9, rtol=0)

    def test_directed_blocks_refine_against_full_chemistry(self):
        matrix = -2 * np.eye(3) + np.diag([0.5, 0.5], -1)
        response = TransportResponse([csr_matrix(matrix)], np.ones(3), np.ones((1, 1)), 400)
        base = np.array([[0.1, 0.2, 0.3]])
        old = np.ones_like(base)
        a = np.full_like(old, 0.3)

        def evaluate(c, m):
            return np.log(2 * c)

        # Emulate small speciation differences between the one-cell kernel
        # and the full backend; only the latter defines the accepted root.
        evaluate.at_cells = lambda c, m, cells: np.log(2 * c[:, cells]) + 1e-5
        result = solve_directed_reactions(
            base,
            old,
            np.zeros_like(old),
            1.0,
            a,
            np.zeros((1, 1)),
            response,
            evaluate,
            np.ones(1, bool),
            ImplicitOptions(absolute_tolerance=1e-12),
        )
        self.assertTrue(evaluate.directed_blocks_used)
        # Independent linear reaction/transport equations for SR = 2*C.
        expected_c = np.linalg.solve(matrix - 0.6 * np.eye(3), matrix @ base[0] - a[0])
        expected_m = old + a * (2 * expected_c - 1)
        np.testing.assert_allclose(result[0], expected_m, atol=1e-12, rtol=0)
        np.testing.assert_allclose(result[1][0], expected_c, atol=1e-12, rtol=0)
        residual = result[0] - old - a * np.expm1(evaluate(result[1], result[0]))
        self.assertLessEqual(np.max(abs(residual)), 1e-12)

    def test_stiff_competing_precipitates_reach_trace_component_root(self):
        nu = np.array([[1.0, 1.0], [0.0, 1.0]])
        response = TransportResponse([csr_matrix([[-0.1]])] * 2, np.ones(1), nu, 400)
        old = np.array([[1.0], [1e-9]])
        base = np.array([[10.0], [1e-5]])
        exact = np.array([[1.5], [1e-5]])
        aqueous = base + nu @ (old - exact)
        b = np.full_like(old, 10.0)
        logk = nu.T @ np.log(aqueous) - np.log1p((exact - old) / b)

        def evaluate(c, m):
            return nu.T @ np.log(c) - logk

        def jacobian(c, m):
            # Deliberately approximate the activity tangent; final acceptance
            # must use the actual chemical residual, including trace potassium.
            return (nu.T @ np.diag(1 / c[:, 0]) @ nu)[:, :, None] * 0.9

        evaluate.jacobian = jacobian
        result = solve_log_reactions(
            base,
            old,
            np.zeros_like(old),
            10.0,
            np.ones_like(old),
            np.zeros((2, 1)),
            response,
            evaluate,
            np.ones(2, bool),
            ImplicitOptions(absolute_tolerance=1e-9, maximum_iterations=80, derivative_refresh=1),
            minimum_amounts=np.array([[1e-12], [1e-9]]),
        )
        np.testing.assert_allclose(result[0], exact, atol=2e-9, rtol=0)
        np.testing.assert_allclose(result[1], aqueous, atol=2e-9, rtol=0)

    def test_augmented_trace_concentrations_preserve_native_mass_equations(self):
        from mf6pqc.coupling.implicit import solve_augmented

        nu = np.array([[1.0, 1.0], [0.0, 1.0]])
        matrix = csr_matrix([[-0.3, 0.2], [0.2, -0.3]])
        response = TransportResponse([matrix, matrix], np.ones(2), nu, 400)
        old = np.array([[1.0, 1.0], [1e-9, 1e-9]])
        exact = np.array([[1.5, 1.3], [1e-5, 9e-6]])
        aqueous = np.array([[9.5, 9.7], [1e-11, 1e-12]])
        dt = 10.0
        base = aqueous - response.apply((old - exact) / dt)[0]
        logk = nu.T @ np.log(aqueous) - np.log1p((exact - old) / dt)

        def evaluate(c, m):
            return nu.T @ np.log(c) - logk

        def tangent(c, m, directions):
            columns = np.array([d for _, _, d in directions]).T
            return np.stack([nu.T @ np.diag(1 / c[:, i]) @ columns for i in range(2)], axis=-1)

        evaluate.component_jacobian = tangent
        result = solve_augmented(
            (
                base,
                old,
                np.zeros_like(old),
                dt,
                np.ones_like(old),
                np.zeros((2, 1)),
                response,
                evaluate,
                np.ones(2, bool),
                ImplicitOptions(absolute_tolerance=1e-8),
            ),
            {"minimum_amounts": np.full_like(old, 1e-9)},
            mineral_guess=exact + 1e-10,
            concentrations_guess=aqueous * 1.0001,
        )
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result[0], exact, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result[1][1], aqueous[1], rtol=1e-7, atol=1e-18)
        residual = matrix @ (result[1] - base).T + (nu @ ((old - result[0]) / dt)).T
        self.assertLess(np.max(abs(residual)), 1e-11)

    def test_local_augmented_trace_concentrations_preserve_mass_equations(self):
        nu = np.array([[1.0, 1.0], [0.0, 1.0]])
        matrix = csr_matrix([[-0.3, 0.2], [0.2, -0.3]])
        response = TransportResponse([matrix, matrix], np.ones(2), nu, 400).local(0)
        old = np.array([[1.0], [0.2]])
        exact = np.array([[1.5], [0.20001]])
        aqueous = np.array([[9.5], [1e-12]])
        dt = 10.0
        base = aqueous - response.apply((old - exact) / dt)[0]
        logk = nu.T @ np.log(aqueous) - np.log1p((exact - old) / dt)

        def evaluate(c, m):
            return nu.T @ np.log(c) - logk

        def tangent(c, m, directions):
            columns = np.array([d for _, _, d in directions]).T
            return (nu.T @ np.diag(1 / c[:, 0]) @ columns)[:, :, None]

        evaluate.component_jacobian = tangent
        evaluate.jacobian = lambda c, m: tangent(c, m, response.directions)
        result = solve_log_reactions(
            base,
            old,
            np.zeros_like(old),
            dt,
            np.ones_like(old),
            np.zeros((2, 1)),
            response,
            evaluate,
            np.ones(2, bool),
            ImplicitOptions(absolute_tolerance=1e-8),
            minimum_amounts=np.full_like(old, 1e-9),
        )
        np.testing.assert_allclose(result[0], exact, atol=1e-10, rtol=0)
        np.testing.assert_allclose(result[1][1], aqueous[1], rtol=1e-7, atol=1e-18)
        residual = -0.3 * (result[1] - base) + nu @ ((old - result[0]) / dt)
        self.assertLess(np.max(abs(residual)), 1e-11)

    def test_fixed_concentration_sources_are_masked_per_component(self):
        matrix = csr_matrix([[-1.0, 0.0], [0.5, -2.0]])
        response = TransportResponse(
            [matrix, matrix],
            np.ones(2),
            np.ones((2, 1)),
            400,
            source_masks=np.array([[False, True], [True, True]]),
        )
        increment, _ = response.apply(np.array([[3.0, 0.0]]))
        np.testing.assert_array_equal(increment[0], 0)
        np.testing.assert_allclose(increment[1], [3.0, 0.75])
        self.assertEqual(len(response.groups), 2)

    def test_invalid_reaction_declarations_fail_early(self):
        for reaction in (
            KineticReaction("X", {"Ca": 1}, -1),
            KineticReaction("X", {"Ca": 1}, None),
            KineticReaction("X", {"Ca": 1}, 1, minimum_amount=None),
            KineticReaction("X", {"Ca": 1}, 1, surface_exponent=1),
            KineticReaction("X", {}, 1),
            KineticReaction("X", {"Ca": 1}, 1, minimum_amount=-1),
        ):
            with self.assertRaises(ConfigurationError):
                ImplicitOptions(reactions=(reaction,)).validated()
        with self.assertRaises(ConfigurationError):
            ImplicitOptions(reactions=(KineticReaction("X", {"Ca": 1}, 1),) * 2).validated()


if __name__ == "__main__":
    unittest.main()
