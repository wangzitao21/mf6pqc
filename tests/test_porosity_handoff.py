"""Conservative pore-volume handoff tests using in-memory backends only.

No native library or example module is loaded. The chemistry double tracks
passive aqueous inventories while a separate matrix mineral changes pore volume.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mf6pqc.coupling.common import (
    commit_reaction_concentrations,
    run_reaction_step,
    update_selected_output,
    write_concentrations_to_modflow,
)
from mf6pqc.coupling.sia import sia_time_step
from mf6pqc.coupling.snia import standard_time_step
from mf6pqc.coupling.strang import strang_time_step
from mf6pqc.coupling.thermal_snia import thermal_time_step
from mf6pqc.exceptions import BackendError, CouplingError
from mf6pqc.feedback import update_medium_properties
from mf6pqc.permeability import KozenyCarmanUpdater


class InventoryChemistry:
    """Store moles per representative litre, not a cached concentration."""

    def __init__(self, porosity, saturation):
        self.porosity = porosity.copy()
        self.saturation = saturation.copy()
        self.moles = None
        self.solution_volume = None
        self.selected = np.array([[-0.5, 1.0, -0.7]])
        self.read_count = 0
        self.reaction_intervals = []
        self.states = {}
        self.time = 0.0
        self.time_step = 0.0

    def SetConcentrations(self, values):
        self.moles = np.asarray(values).reshape(-1, 3) * self.porosity * self.saturation

    def GetConcentrations(self):
        self.read_count += 1
        volume = (
            self.porosity * self.saturation
            if self.solution_volume is None
            else self.solution_volume
        )
        return (self.moles / volume).ravel()

    def SetPorosity(self, values):
        self.porosity = values.copy()

    def SetTemperature(self, values):
        self.temperature = values.copy()

    def SetTime(self, value):
        self.time = value

    def SetTimeStep(self, value):
        self.time_step = value

    def RunCells(self):
        self.reaction_intervals.append((self.time, self.time_step))

    def GetSelectedOutput(self):
        return self.selected.ravel().copy()

    def GetDensityCalculated(self):
        return np.array([1.1, 1.2, 1.3])

    def StateSave(self, identifier):
        self.states[identifier] = self.moles.copy()

    def StateApply(self, identifier):
        self.moles = self.states[identifier].copy()

    def StateDelete(self, identifier):
        del self.states[identifier]


class RecordingTransport:
    """Zero-flux transport that records the inventory supplied to each solve."""

    def __init__(self, sim, state):
        self.sim = sim
        self.state = state
        self.current_time = 0.0
        self.dt = 0.0
        self.inventories = []

    def prepare_time_step(self, dt):
        self.dt = dt

    def prepare_solve(self, solution_id):
        pass

    def solve(self, solution_id):
        self.inventories.append(transport_inventory(self.sim, self.state))
        return True

    def finalize_solve(self, solution_id):
        pass

    def finalize_time_step(self):
        self.current_time += self.dt

    def get_current_time(self):
        return self.current_time


def packed_concentrations(state):
    return np.concatenate([info["ptr"].ravel() for info in state.concentration_variables.values()])


def transport_inventory(sim, state):
    return np.stack(
        [
            info["ptr"].ravel() * sim.thetam_ptrs[name] * sim.saturation * state.bulk_cell_volume
            for name, info in state.concentration_variables.items()
        ]
    )


def make_case(*, feedback=True, thermal=False, density=False):
    porosity = np.array([0.2, 0.4, 0.3])
    saturation = np.array([1.0, 0.5, 0.8])
    components = ["Tracer", "Charge", "H2O"]
    initial = np.array([[1.0, 2.0, 4.0], [-0.01, 0.02, -0.03], [55.0, 54.0, 53.0]])
    chemistry = InventoryChemistry(porosity, saturation)
    chemistry.SetConcentrations(initial.ravel())
    sim = SimpleNamespace(
        nxyz=3,
        ncomps=3,
        components=components,
        signed_components=frozenset({"charge"}),
        phreeqc_rm=chemistry,
        porosity=porosity.copy(),
        saturation=saturation.copy(),
        porosity_update_mask=np.array([True, True, False]),
        if_update_porosity_K=feedback,
        if_update_density=density,
        use_phreeqc_calculated_density=True,
        density_ptr=np.zeros(3),
        k_update_density_prev=chemistry.GetDensityCalculated(),
        k_update_viscosity_prev=np.ones(3),
        viscosity=np.ones(3),
        if_update_diffc=False,
        output_indices=np.array([0]),
        mineral_volumes=np.array([[0.1]]),
        headings=["d_Matrix"],
        selected_output=np.zeros((1, 3)),
        thetam_ptrs={name: porosity.copy() for name in components},
        _update_K=KozenyCarmanUpdater().update,
        save_interval=1,
        save_interval_offset=0,
        save_steps=None,
        reaction_steps=None,
        progress_interval=1000,
        results=[np.zeros((1, 3))],
        results_porosity=[porosity.copy()],
        results_K=[np.ones(3)],
        result_times=[0.0],
        fail_on_nonconvergence=True,
        modflow_convergence_failures=[],
        has_water_only_sinks=False,
        water_only_sink_rates=np.zeros(3),
        sia_max_iterations=4,
        sia_rtol=1e-10,
        sia_atol=1e-12,
        sia_source_relaxation=1.0,
        sia_density_relaxation=1.0,
        sia_fail_on_nonconvergence=True,
        sia_rate_evaluator=None,
        sia_iterations=[],
        sia_diagnostics=[],
        sia_convergence_failures=[],
        energy_enabled=thermal,
        vsc_enabled=False,
        sync_gwe_temperature_to_phreeqc=True,
        energy_binding=SimpleNamespace(
            temperature_ptr=np.array([30.0, 40.0, 50.0]),
            est_porosity_ptr=porosity.copy(),
        ),
        results_temperature=[np.array([30.0, 40.0, 50.0])],
        results_temperature_for_flow=[np.array([30.0, 40.0, 50.0])],
    )
    state = SimpleNamespace(
        concentration_variables={
            name: {"ptr": initial[index].copy()} for index, name in enumerate(components)
        },
        species_slices=tuple(slice(i * 3, (i + 1) * 3) for i in range(3)),
        transported=np.empty(9),
        reacted=initial.ravel().copy(),
        reaction_input=np.empty(9),
        previous_time_concentrations=initial.ravel().copy(),
        previous_iteration_concentrations=initial.ravel().copy(),
        source_rates=np.zeros(9),
        candidate_source_rates=np.zeros(9),
        concentration_difference=np.empty(9),
        coupling_difference=np.empty(9),
        source_difference=np.empty(9),
        solution_iterations={1: np.array([1])},
        source_variables={name: {"ptr": np.zeros(3)} for name in components},
        water_sink_sources=None,
        bulk_cell_volume=np.array([2.0, 3.0, 4.0]),
        mobile_water_volume=np.array([2.0, 3.0, 4.0]) * porosity * saturation,
        current_time=0.0,
        end_time=2.0,
        logical_step=0,
        transport_step=0,
        last_reaction_time=0.0,
        current_k11=np.ones(3) if feedback else None,
        previous_density=None,
        candidate_density=None,
        density_difference=None,
        picard_iteration=0,
        time_step_schedule=np.ones(2),
        current_dt=0.0,
    )
    sim.modflow_api = RecordingTransport(sim, state)
    return sim, state, initial


class PorosityHandoffTests(unittest.TestCase):
    def check_two_steps(self, module, step, *, thermal=False, strang=False):
        sim, state, initial = make_case(thermal=thermal)
        expected_inventory = transport_inventory(sim, state)
        if strang:
            state.time_step_schedule = np.full(4, 0.5)
        # NPF dirty flags are unrelated to concentration handoff. All chemistry,
        # porosity updates, concentration copies, and SIA iteration logic run.
        with patch(f"mf6pqc.coupling.{module}.write_conductivity_for_step"):
            for expected_porosity in ([0.25, 0.3, 0.3], [0.3, 0.2, 0.3]):
                step(sim, state)
                np.testing.assert_allclose(sim.porosity, expected_porosity)
                np.testing.assert_allclose(transport_inventory(sim, state), expected_inventory)
                np.testing.assert_array_equal(packed_concentrations(state), state.reacted)
                np.testing.assert_array_equal(state.reacted.reshape(3, 3)[:, 2], initial[:, 2])
                np.testing.assert_array_equal(sim.results[-1], sim.phreeqc_rm.selected)
                for pointer in sim.thetam_ptrs.values():
                    np.testing.assert_array_equal(pointer, sim.porosity)
        # Also catch a stale concentration passed to the next transport solve,
        # particularly Strang's second half-step within the same logical step.
        for inventory in sim.modflow_api.inventories:
            np.testing.assert_allclose(inventory, expected_inventory)
        np.testing.assert_array_equal(sim.result_times, [0.0, 1.0, 2.0])
        self.assertEqual(len(sim.results), 3)
        self.assertEqual(len(sim.results_porosity), 3)
        self.assertEqual(sim.phreeqc_rm.states, {})
        return sim, state

    def test_snia_preserves_inventory_through_dissolution_precipitation_and_mask(self):
        sim, _ = self.check_two_steps("snia", standard_time_step)
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 2)

    def test_strang_second_half_step_receives_conservative_concentrations(self):
        sim, _ = self.check_two_steps("strang", strang_time_step, strang=True)
        durations = [dt for _, dt in sim.phreeqc_rm.reaction_intervals]
        self.assertEqual(durations, [86400.0, 0.0, 86400.0, 0.0])

    def test_sia_commits_new_volume_after_convergence(self):
        sim, state = self.check_two_steps("sia", sia_time_step)
        self.assertEqual(sim.sia_iterations, [2, 2])
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 4)
        self.assertEqual(sim.sia_convergence_failures, [])
        np.testing.assert_array_equal(
            state.mobile_water_volume, state.bulk_cell_volume * sim.porosity * sim.saturation
        )

    def test_thermal_snia_preserves_inventory_and_updates_est_porosity(self):
        sim, _ = self.check_two_steps("thermal_snia", thermal_time_step, thermal=True)
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 2)
        self.assertEqual(len(sim.results_temperature), 3)
        np.testing.assert_array_equal(sim.energy_binding.est_porosity_ptr, sim.porosity)
        np.testing.assert_array_equal(sim.phreeqc_rm.temperature, [30.0, 40.0, 50.0])

    def test_fixed_porosity_density_feedback_keeps_original_endpoint(self):
        sim, state, initial = make_case(feedback=False, density=True)
        standard_time_step(sim, state)
        np.testing.assert_array_equal(packed_concentrations(state), initial.ravel())
        np.testing.assert_array_equal(sim.porosity, [0.2, 0.4, 0.3])
        np.testing.assert_array_equal(sim.density_ptr, [1100.0, 1200.0, 1300.0])
        self.assertEqual(sim.phreeqc_rm.read_count, 1)
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 1)

    def test_stateless_sia_does_not_read_back_a_diagnostic_chemistry_state(self):
        sim, state, initial = make_case(feedback=False)
        sim.sia_rate_evaluator = lambda _components, values, _time: np.zeros_like(values)
        sia_time_step(sim, state)
        np.testing.assert_array_equal(packed_concentrations(state), initial.ravel())
        self.assertEqual(sim.phreeqc_rm.read_count, 0)
        self.assertEqual(sim.phreeqc_rm.reaction_intervals, [(86400.0, 0.0)])

    def test_backend_solution_volume_convention_is_not_manually_rescaled(self):
        sim, state, _ = make_case()
        run_reaction_step(sim, state.reacted.copy(), state.reacted, 0.0, 1.0)
        sim.phreeqc_rm.solution_volume = np.array([0.18, 0.22, 0.25])
        update_selected_output(sim)
        state.current_k11 = update_medium_properties(sim, state.current_k11, 0)
        expected = (sim.phreeqc_rm.moles / sim.phreeqc_rm.solution_volume).ravel()
        commit_reaction_concentrations(sim, state)
        np.testing.assert_array_equal(state.reacted, expected)
        np.testing.assert_array_equal(packed_concentrations(state), expected)
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 1)

    def test_existing_property_update_then_readback_is_idempotent(self):
        sim, state, _ = make_case()
        expected_inventory = transport_inventory(sim, state)
        run_reaction_step(sim, state.reacted.copy(), state.reacted, 0.0, 1.0)
        update_selected_output(sim)
        state.current_k11 = update_medium_properties(sim, state.current_k11, 0)
        # Preserve the existing custom-loop contract without importing examples.
        state.reacted[:] = sim.phreeqc_rm.GetConcentrations()
        write_concentrations_to_modflow(
            state.concentration_variables, state.species_slices, state.reacted
        )
        expected = packed_concentrations(state)
        commit_reaction_concentrations(sim, state)
        commit_reaction_concentrations(sim, state)
        np.testing.assert_array_equal(packed_concentrations(state), expected)
        np.testing.assert_allclose(transport_inventory(sim, state), expected_inventory)
        self.assertEqual(len(sim.phreeqc_rm.reaction_intervals), 1)

    def test_invalid_readback_does_not_overwrite_concentrations(self):
        for bad, error in (
            (np.ones(8), BackendError),
            (np.full(9, np.nan), CouplingError),
            (np.full(9, np.inf), CouplingError),
        ):
            with self.subTest(error=error.__name__, values=bad):
                sim, state, initial = make_case()
                sim.phreeqc_rm.GetConcentrations = lambda values=bad: values
                with self.assertRaises(error):
                    commit_reaction_concentrations(sim, state)
                np.testing.assert_array_equal(state.reacted, initial.ravel())
                np.testing.assert_array_equal(packed_concentrations(state), initial.ravel())


if __name__ == "__main__":
    unittest.main()
