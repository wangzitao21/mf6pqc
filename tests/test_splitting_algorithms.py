from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from mf6pqc.coupling.strang import solve_transport_substep, strang_time_step


class StrangOrderingTests(unittest.TestCase):
    def test_midpoint_conductivity_is_written_after_prepare_time_step(self) -> None:
        events: list[tuple] = []

        class Api:
            def prepare_time_step(self, dt):
                events.append(("prepare", dt))

            def finalize_time_step(self):
                events.append(("finalize",))

            def get_current_time(self):
                return 1.5

        sim = SimpleNamespace(modflow_api=Api())
        state = SimpleNamespace(current_time=1.0)

        def write_k(_sim, _k, logical_step, *, force=False):
            events.append(("write_k", logical_step, force))

        def solve(_sim, _state, density):
            events.append(("solve", density))

        with (
            patch(
                "mf6pqc.coupling.strang.write_conductivity_for_step",
                side_effect=write_k,
            ),
            patch(
                "mf6pqc.coupling.strang.solve_modflow_solutions",
                side_effect=solve,
            ),
        ):
            solve_transport_substep(
                sim,
                state,
                0.5,
                None,
                np.array([2.0]),
                0,
                force_conductivity_write=True,
            )

        self.assertEqual(
            events,
            [
                ("prepare", 0.5),
                ("write_k", 0, True),
                ("solve", None),
                ("finalize",),
            ],
        )
        self.assertEqual(state.current_time, 1.5)

    def test_reaction_and_property_feedback_are_centered_between_half_steps(self) -> None:
        events: list[tuple] = []
        concentration = np.array([1.0])
        state = SimpleNamespace(
            transport_step=0,
            time_step_schedule=np.array([0.5, 0.5]),
            current_time=2.0,
            current_k11=np.array([10.0]),
            logical_step=0,
            concentration_variables={"A": {"ptr": concentration}},
            species_slices=(slice(0, 1),),
            transported=np.empty(1),
            reacted=np.empty(1),
            end_time=3.0,
        )
        sim = SimpleNamespace(
            if_update_density=False,
            components=["A"],
            signed_components=frozenset(),
            progress_interval=100,
        )

        def solve_substep(
            _sim,
            current_state,
            dt,
            _density,
            current_k11,
            _logical_step,
            *,
            force_conductivity_write=False,
        ):
            events.append(
                (
                    "transport",
                    dt,
                    float(current_k11[0]),
                    force_conductivity_write,
                )
            )
            current_state.current_time += dt

        def read_concentrations(_variables, _slices, destination):
            destination[:] = concentration

        def react(_sim, transported, reacted, start_time, dt):
            events.append(("reaction", start_time, dt))
            reacted[:] = transported * 0.5

        def write_concentrations(_variables, _slices, source):
            concentration[:] = source

        def update_properties(_sim, current_k11, logical_step):
            events.append(("properties", logical_step))
            return current_k11 * 0.5

        def synchronize(_sim, values, current_time, **_kwargs):
            events.append(("synchronize", current_time, float(values[0])))

        with (
            patch(
                "mf6pqc.coupling.strang.solve_transport_substep",
                side_effect=solve_substep,
            ),
            patch(
                "mf6pqc.coupling.strang.read_concentrations_from_modflow",
                side_effect=read_concentrations,
            ),
            patch("mf6pqc.coupling.strang.enforce_component_domains"),
            patch(
                "mf6pqc.coupling.strang.run_reaction_step",
                side_effect=react,
            ),
            patch("mf6pqc.coupling.strang.update_selected_output"),
            patch(
                "mf6pqc.coupling.strang.write_concentrations_to_modflow",
                side_effect=write_concentrations,
            ),
            patch(
                "mf6pqc.coupling.strang.update_medium_properties",
                side_effect=update_properties,
            ),
            patch(
                "mf6pqc.coupling.strang.synchronize_phreeqcrm_solution",
                side_effect=synchronize,
            ),
            patch("mf6pqc.coupling.strang.save_time_step_results"),
            patch("mf6pqc.coupling.strang.log_progress"),
        ):
            strang_time_step(sim, state)

        self.assertEqual(
            events[:4],
            [
                ("transport", 0.5, 10.0, False),
                ("reaction", 2.0, 1.0),
                ("properties", 0),
                ("transport", 0.5, 5.0, True),
            ],
        )
        self.assertEqual(state.current_time, 3.0)
        self.assertEqual(state.logical_step, 1)
        self.assertEqual(state.transport_step, 2)


if __name__ == "__main__":
    unittest.main()
