from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from mf6pqc.constants import SECONDS_PER_DAY
from mf6pqc.coupling.common import (
    build_time_step_schedule,
    enforce_component_domains,
    read_concentrations_from_modflow,
    run_reaction_step,
    should_save_time_step,
    should_run_reaction,
    solve_modflow_solutions,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.sia import (
    build_reaction_input,
    check_picard_convergence,
    update_sources,
    update_sources_from_instantaneous_rates,
)
from mf6pqc.coupling.strang import validate_strang_schedule
from mf6pqc.exceptions import CouplingError


class TdisApi:
    def __init__(self, perlen, nstp, tsmult) -> None:
        self.values = {
            "__INPUT__/SIM/TDIS/PERLEN": np.asarray(perlen),
            "__INPUT__/SIM/TDIS/NSTP": np.asarray(nstp),
            "__INPUT__/SIM/TDIS/TSMULT": np.asarray(tsmult),
        }

    def get_value(self, address):
        return self.values[address]


class RecordingReactionBackend:
    def __init__(self, result) -> None:
        self.result = np.asarray(result, dtype=float)
        self.concentrations = None
        self.time = None
        self.time_step = None
        self.run_count = 0

    def SetConcentrations(self, values) -> None:
        self.concentrations = np.asarray(values).copy()

    def SetTime(self, value) -> None:
        self.time = value

    def SetTimeStep(self, value) -> None:
        self.time_step = value

    def RunCells(self) -> None:
        self.run_count += 1

    def GetConcentrations(self):
        return self.result.copy()


class NonConvergingApi:
    def __init__(self) -> None:
        self.prepared = []
        self.finalized = []
        self.solve_calls = 0

    def get_subcomponent_count(self) -> int:
        return 1

    def prepare_solve(self, solution_id: int) -> None:
        self.prepared.append(solution_id)

    def solve(self, solution_id: int) -> bool:
        self.solve_calls += 1
        return False

    def finalize_solve(self, solution_id: int) -> None:
        self.finalized.append(solution_id)

    def get_current_time(self) -> float:
        return 2.5


class ScheduleTests(unittest.TestCase):
    def test_equal_and_geometric_periods_are_expanded(self) -> None:
        api = TdisApi([10.0, 7.0], [2, 3], [1.0, 2.0])
        np.testing.assert_allclose(build_time_step_schedule(api), [5, 5, 1, 2, 4])

    def test_schedule_rejects_nonphysical_period_data(self) -> None:
        api = TdisApi([1.0], [0], [1.0])
        with self.assertRaisesRegex(ValueError, "positive"):
            build_time_step_schedule(api)

    def test_strang_schedule_returns_logical_pair_durations(self) -> None:
        np.testing.assert_allclose(
            validate_strang_schedule(np.array([0.1, 0.1, 0.2, 0.2])),
            [0.2, 0.4],
        )

    def test_strang_schedule_rejects_unequal_pair_before_advancing(self) -> None:
        with self.assertRaisesRegex(CouplingError, "logical step 2"):
            validate_strang_schedule(np.array([0.1, 0.1, 0.2, 0.25]))


class ConcentrationTransferTests(unittest.TestCase):
    def test_charge_component_keeps_its_signed_domain(self) -> None:
        values = np.array([-2.0, 3.0, -4.0, 5.0])
        enforce_component_domains(
            values,
            ["Na", "Charge"],
            [slice(0, 2), slice(2, 4)],
        )
        self.assertGreater(values[0], 0.0)
        self.assertEqual(values[1], 3.0)
        np.testing.assert_array_equal(values[2:], [-4.0, 5.0])

    def test_signed_domain_can_be_disabled_for_legacy_compatibility(self) -> None:
        values = np.array([-2.0])
        enforce_component_domains(
            values,
            ["Charge"],
            [slice(0, 1)],
            signed_components=(),
        )
        self.assertGreater(values[0], 0.0)

    def test_component_major_buffers_round_trip(self) -> None:
        first = np.array([1.0, 2.0])
        second = np.array([3.0, 4.0])
        info = {"A": {"ptr": first}, "B": {"ptr": second}}
        slices = [slice(0, 2), slice(2, 4)]
        packed = np.empty(4)
        read_concentrations_from_modflow(info, slices, packed)
        np.testing.assert_array_equal(packed, [1, 2, 3, 4])

        write_concentrations_to_modflow(info, slices, packed + 10.0)
        np.testing.assert_array_equal(first, [11, 12])
        np.testing.assert_array_equal(second, [13, 14])

    def test_reaction_step_uses_seconds_for_phreeqcrm(self) -> None:
        backend = RecordingReactionBackend([5.0, 6.0])
        sim = SimpleNamespace(phreeqc_rm=backend)
        output = np.empty(2)
        run_reaction_step(sim, np.array([1.0, 2.0]), output, 3.0, 0.25)
        np.testing.assert_array_equal(backend.concentrations, [1.0, 2.0])
        self.assertEqual(backend.time, 3.0 * SECONDS_PER_DAY)
        self.assertEqual(backend.time_step, 0.25 * SECONDS_PER_DAY)
        np.testing.assert_array_equal(output, [5.0, 6.0])

    def test_reaction_step_rejects_nonfinite_interval_start(self) -> None:
        backend = RecordingReactionBackend([5.0])
        sim = SimpleNamespace(phreeqc_rm=backend)
        with self.assertRaisesRegex(CouplingError, "start time"):
            run_reaction_step(
                sim, np.array([1.0]), np.empty(1), float("nan"), 0.25
            )
        self.assertEqual(backend.run_count, 0)


class OutputScheduleTests(unittest.TestCase):
    def test_legacy_interval_and_offset_semantics_are_preserved(self) -> None:
        sim = SimpleNamespace(save_steps=None, save_interval=3, save_interval_offset=1)
        self.assertFalse(should_save_time_step(sim, 0))
        self.assertFalse(should_save_time_step(sim, 1))
        self.assertTrue(should_save_time_step(sim, 2))

    def test_explicit_steps_are_one_based(self) -> None:
        sim = SimpleNamespace(save_steps=frozenset({1, 4}), save_interval=99)
        self.assertTrue(should_save_time_step(sim, 0))
        self.assertFalse(should_save_time_step(sim, 1))
        self.assertTrue(should_save_time_step(sim, 3))

    def test_reaction_schedule_defaults_to_every_transport_step(self) -> None:
        sim = SimpleNamespace(reaction_steps=None)
        self.assertTrue(should_run_reaction(sim, 0))
        self.assertTrue(should_run_reaction(sim, 10))

    def test_explicit_reaction_steps_are_one_based(self) -> None:
        sim = SimpleNamespace(reaction_steps=frozenset({4, 8}))
        self.assertFalse(should_run_reaction(sim, 0))
        self.assertTrue(should_run_reaction(sim, 3))
        self.assertTrue(should_run_reaction(sim, 7))


class SolverFailureTests(unittest.TestCase):
    def _sim(self, fail: bool):
        return SimpleNamespace(
            modflow_api=NonConvergingApi(),
            if_update_density=False,
            modflow_convergence_failures=[],
            fail_on_nonconvergence=fail,
            water_only_sink_rates=np.zeros(1),
        )

    @staticmethod
    def _state():
        return SimpleNamespace(
            solution_iterations={1: np.array([3])},
            water_sink_sources=None,
        )

    def test_nonconvergence_is_recorded(self) -> None:
        sim = self._sim(False)
        solve_modflow_solutions(sim, self._state(), None)
        self.assertEqual(sim.modflow_api.solve_calls, 3)
        self.assertEqual(sim.modflow_convergence_failures[0]["solution_id"], 1)

    def test_strict_mode_raises_after_recording(self) -> None:
        sim = self._sim(True)
        with self.assertRaisesRegex(RuntimeError, "failed to converge"):
            solve_modflow_solutions(sim, self._state(), None)
        self.assertEqual(len(sim.modflow_convergence_failures), 1)


class SiaConvergenceTests(unittest.TestCase):
    @staticmethod
    def _sim():
        return SimpleNamespace(sia_atol=1.0e-8, sia_rtol=1.0e-4)

    @staticmethod
    def _state():
        transported = np.array([1.0, 2.0])
        source_rates = np.array([0.1, 0.2])
        return SimpleNamespace(
            picard_iteration=1,
            transported=transported,
            reaction_input=transported.copy(),
            reacted=transported.copy(),
            previous_iteration_concentrations=np.array([1.0, 2.0]),
            concentration_difference=np.empty(2),
            coupling_difference=np.zeros(2),
            candidate_density=None,
            previous_density=None,
            density_difference=None,
            source_rates=source_rates,
            candidate_source_rates=source_rates.copy(),
            source_difference=np.zeros(2),
            mobile_water_volume=np.array([0.25, 0.25]),
            current_dt=0.5,
            species_slices=(slice(0, 2),),
        )

    def test_unchanged_concentration_and_sources_converge(self) -> None:
        self.assertTrue(check_picard_convergence(self._sim(), self._state()))

    def test_first_iteration_never_converges(self) -> None:
        state = self._state()
        state.picard_iteration = 0
        self.assertFalse(check_picard_convergence(self._sim(), state))

    def test_reaction_source_rate_uses_mobile_water_volume(self) -> None:
        ptr = np.zeros(2)
        state = self._state()
        state.reacted = np.array([3.0, 5.0])
        state.transported = np.array([1.0, 1.0])
        state.reaction_input = np.array([1.0, 1.0])
        state.source_rates = np.zeros(2)
        state.candidate_source_rates = np.zeros(2)
        state.coupling_difference = np.empty(2)
        state.source_difference = np.empty(2)
        state.source_variables = {"A": {"ptr": ptr}}
        state.concentration_variables = {"A": {"ptr": np.ones(2)}}
        sim = SimpleNamespace(
            components=["A"],
            sia_source_relaxation=1.0,
            water_only_sink_rates=np.zeros(2),
        )
        update_sources(sim, state, dt=0.5)
        np.testing.assert_allclose(ptr, [1.0, 2.0])

    def test_reaction_input_removes_the_applied_reaction_source(self) -> None:
        state = self._state()
        state.transported = np.array([0.8, 1.6])
        state.source_rates = np.array([-0.1, -0.2])
        state.mobile_water_volume = np.array([0.5, 0.5])
        state.source_difference = np.empty(2)
        build_reaction_input(state, dt=1.0)
        np.testing.assert_allclose(state.reaction_input, [1.0, 2.0])

    def test_source_is_evaluated_from_reaction_base_not_modflow_endpoint(self) -> None:
        ptr = np.zeros(1)
        state = SimpleNamespace(
            reacted=np.array([0.67]),
            transported=np.array([0.8]),
            reaction_input=np.array([1.0]),
            source_rates=np.array([-0.2]),
            candidate_source_rates=np.empty(1),
            source_difference=np.empty(1),
            coupling_difference=np.empty(1),
            mobile_water_volume=np.ones(1),
            species_slices=(slice(0, 1),),
            source_variables={"A": {"ptr": ptr}},
            concentration_variables={"A": {"ptr": np.ones(1)}},
        )
        sim = SimpleNamespace(
            components=["A"],
            sia_source_relaxation=1.0,
            water_only_sink_rates=np.zeros(1),
        )
        update_sources(sim, state, dt=1.0)
        np.testing.assert_allclose(state.candidate_source_rates, [-0.33])
        np.testing.assert_allclose(state.source_difference, [-0.13])

    def test_instantaneous_rate_callback_is_converted_to_mass_rate(self) -> None:
        pointers = {"A": np.zeros(2), "B": np.zeros(2)}
        state = SimpleNamespace(
            transported=np.array([1.0, 2.0, 3.0, 4.0]),
            reacted=np.empty(4),
            source_rates=np.zeros(4),
            candidate_source_rates=np.empty(4),
            source_difference=np.empty(4),
            coupling_difference=np.empty(4),
            mobile_water_volume=np.array([0.5, 2.0]),
            species_slices=(slice(0, 2), slice(2, 4)),
            source_variables={
                name: {"ptr": pointer} for name, pointer in pointers.items()
            },
            concentration_variables={
                name: {"ptr": np.ones(2)} for name in pointers
            },
        )

        def evaluator(components, concentrations, target_time):
            self.assertEqual(components, ("A", "B"))
            self.assertFalse(concentrations.flags.writeable)
            self.assertEqual(target_time, 2.0)
            return np.array([[-1.0, -2.0], [3.0, 4.0]])

        sim = SimpleNamespace(
            components=["A", "B"],
            ncomps=2,
            nxyz=2,
            sia_rate_evaluator=evaluator,
            sia_source_relaxation=1.0,
            water_only_sink_rates=np.zeros(2),
        )
        update_sources_from_instantaneous_rates(sim, state, target_time=2.0)
        np.testing.assert_allclose(pointers["A"], [-0.5, -4.0])
        np.testing.assert_allclose(pointers["B"], [1.5, 8.0])
        np.testing.assert_allclose(state.reacted, state.transported)
        np.testing.assert_allclose(state.coupling_difference, 0.0)

    def test_instantaneous_rate_callback_shape_is_validated(self) -> None:
        state = self._state()
        state.source_variables = {"A": {"ptr": np.zeros(2)}}
        state.concentration_variables = {"A": {"ptr": np.ones(2)}}
        sim = SimpleNamespace(
            components=["A"],
            ncomps=1,
            nxyz=2,
            sia_rate_evaluator=lambda *_: np.zeros((1, 1)),
            sia_source_relaxation=1.0,
            water_only_sink_rates=np.zeros(2),
        )
        with self.assertRaisesRegex(CouplingError, "returned shape"):
            update_sources_from_instantaneous_rates(
                sim, state, target_time=1.0
            )

    def test_transport_reaction_closure_is_required(self) -> None:
        state = self._state()
        state.reacted = state.transported + 1.0e-3
        state.coupling_difference[:] = state.reacted - state.transported
        self.assertFalse(check_picard_convergence(self._sim(), state))

    def test_unrelaxed_source_residual_is_required(self) -> None:
        state = self._state()
        state.candidate_source_rates = state.source_rates + 1.0e-2
        state.source_difference[:] = 1.0e-2
        self.assertFalse(check_picard_convergence(self._sim(), state))


if __name__ == "__main__":
    unittest.main()
