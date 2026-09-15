import gc
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mf6pqc import (
    MF6PQC,
    BackendPaths,
    CellFields,
    CouplingHooks,
    CouplingMethod,
    SimulationConfig,
)
from mf6pqc.backends import CheckedPhreeqcRM, NativeBackendFactory, validate_modflow_workspace
from mf6pqc.coupling.common import (
    build_nonnegative_slices,
    enforce_component_domains,
    update_selected_output,
)
from mf6pqc.exceptions import BackendError, ConfigurationError, CouplingError, PropertyUpdateError
from mf6pqc.feedback import setup_boundary_conductance_updates, update_medium_properties
from mf6pqc.input_processing import create_ic_array_from_map
from mf6pqc.parallel import _ProcessPhreeqcRM
from mf6pqc.properties import get_calculated_density
from mf6pqc.results import FrameBuffer, ProgressReporter, ResultHistory, prepare_results
from tests.test_parallel_backend import Reactor
from tests.test_porosity_handoff import make_case
from tests.test_public_api import FakeFactory


class ConfigurationOwnershipTests(unittest.TestCase):
    def test_structured_config_initializes_without_legacy_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "db").touch()
            (directory / "input").touch()
            porosity = np.array([0.2, 0.4])
            config = SimulationConfig(
                "grouped",
                2,
                BackendPaths(directory / "db", directory / "input", None, None, directory / "out"),
                fields=CellFields(porosity=porosity),
                backend_factory=FakeFactory(),
            )
            with (
                patch.object(
                    SimulationConfig,
                    "to_legacy_kwargs",
                    side_effect=AssertionError("legacy translation used"),
                ),
                MF6PQC.from_config(config) as sim,
            ):
                porosity[:] = 0.9
                np.testing.assert_array_equal(sim.cells.porosity, [0.2, 0.4])
                sim.setup({"solution": 0})
                self.assertIs(sim.phreeqc_rm, sim.chemistry.backend)
                self.assertIs(sim.results, sim.history.results)
                self.assertIsNot(sim.config, config)
                sim.porosity = np.array([0.3, 0.5])
                np.testing.assert_array_equal(sim.cells.porosity, [0.3, 0.5])
                self.assertEqual(
                    set(sim.__dict__),
                    {"config", "cells", "chemistry", "transport", "lifecycle", "history"},
                )

    def test_fractional_boundary_indices_fail_before_native_creation(self):
        for value in (1.9, True, np.nan, 3, -4):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                MF6PQC(
                    nxyz=3,
                    if_update_porosity_K=True,
                    boundary_conductance_updates={"GHB": {"cell_index": value, "distance": 1.0}},
                )
        api = SimpleNamespace(
            get_var_address=lambda *args: "COND", get_value_ptr=lambda address: np.ones(1)
        )
        sim = SimpleNamespace(
            nxyz=3,
            flow_model_name="GWF",
            modflow_api=api,
            boundary_conductance_updates={"GHB": {"cell_index": 1.9, "distance": 1.0}},
        )
        with self.assertRaises(ConfigurationError):
            setup_boundary_conductance_updates(sim)

    def test_modflow_variable_names_are_cached_only_while_loading(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                reads = []

                def read_names(reads=reads):
                    reads.append(None)
                    return ("GWF/X",)

                api = SimpleNamespace(get_input_var_names=read_names)

                def load(actual, fail=fail):
                    for _ in range(5):
                        self.assertEqual(actual.get_input_var_names(), ("GWF/X",))
                    if fail:
                        raise RuntimeError("load failed")
                    return actual

                fake = SimpleNamespace(
                    extensions=SimpleNamespace(ApiSimulation=SimpleNamespace(load=load))
                )
                with patch.dict("sys.modules", {"modflowapi": fake}):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "load failed"):
                            NativeBackendFactory().load_modflow_simulation(api)
                    else:
                        self.assertIs(NativeBackendFactory().load_modflow_simulation(api), api)
                self.assertEqual(len(reads), 1)
                self.assertIs(api.get_input_var_names, read_names)
                api.get_input_var_names()
                self.assertEqual(len(reads), 2)

    def test_density_is_resolved_by_heading_independently_of_column_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "db").touch()
            (directory / "input").touch()
            factory = FakeFactory()
            with MF6PQC(
                nxyz=2,
                db_path=directory / "db",
                pqi_path=directory / "input",
                output_dir=directory,
                if_update_density=True,
                backend_factory=factory,
            ) as sim:
                factory.chemistry.GetSelectedOutputHeadings = lambda: ["RHO", "A"]
                factory.chemistry.GetSelectedOutput = lambda: np.array([1.1, 1.2, 0.25, 0.5])
                sim.setup({"solution": 0})
                np.testing.assert_allclose(get_calculated_density(sim), [1100.0, 1200.0])
                np.testing.assert_array_equal(sim.selected_output[1], [0.25, 0.5])


class ResultHistoryTests(unittest.TestCase):
    def test_memory_and_disk_buffers_preserve_snapshots_and_live_views(self):
        for storage in ("memory", "disk"):
            with self.subTest(storage=storage):
                buffer = FrameBuffer(3, (2,), [[1.0, 2.0]], storage=storage)
                source = np.array([3.0, 4.0])
                buffer.append(source)
                source[:] = 99
                view = np.asarray(buffer)
                buffer.append([5.0, 6.0])
                np.testing.assert_array_equal(view, [[1, 2], [3, 4]])
                np.testing.assert_array_equal(np.asarray(buffer), [[1, 2], [3, 4], [5, 6]])
                with self.assertRaises(CouplingError):
                    buffer.append([7, 8])
                del buffer
                gc.collect()
                np.testing.assert_array_equal(view, [[1, 2], [3, 4]])

    def test_sparse_reaction_schedule_allocates_only_saved_frames(self):
        history = ResultHistory(results=[np.ones((2, 3))], result_times=[0.0])
        sim = SimpleNamespace(
            history=history,
            nxyz=3,
            headings=["A", "B"],
            save_steps=None,
            save_interval=2,
            save_interval_offset=1,
            reaction_steps=frozenset({2, 4, 6}),
            if_update_porosity_K=False,
            if_update_diffc=True,
            energy_enabled=False,
            result_storage="disk",
        )
        prepare_results(sim, SimpleNamespace(), 6)
        self.assertEqual(history.results.values.shape, (4, 2, 3))
        self.assertEqual(history.results_diffc.values.shape, (3, 3))
        self.assertEqual(len(history.results), 1)
        self.assertEqual(len(history.results_diffc), 0)
        history.results.append(np.full((2, 3), 2.0))
        np.testing.assert_array_equal(
            np.asarray(history.results), [np.ones((2, 3)), np.full((2, 3), 2.0)]
        )

    def test_invalid_storage_and_out_of_range_saves_are_rejected(self):
        with self.assertRaises(ConfigurationError):
            MF6PQC(result_storage="unknown")
        sim = SimpleNamespace(save_steps=frozenset({4}))
        with self.assertRaises(CouplingError):
            prepare_results(sim, SimpleNamespace(), 3)

    def test_progress_includes_start_intermediate_and_final_without_duplicates(self):
        state = SimpleNamespace(logical_step=0, current_time=0.0)
        reporter = ProgressReporter(55.0, 55, 1000, clock=lambda: 0.0)
        with self.assertLogs("mf6pqc.results", level="INFO") as logs:
            for step in range(56):
                state.logical_step = step
                state.current_time = float(step)
                reporter.report(state)
            reporter.report(state)
        self.assertIn("step=0/55", logs.output[0])
        self.assertIn("step=55/55 (100.0%)", logs.output[-1])
        self.assertEqual(len(logs.output), 13)

    def test_slow_steps_trigger_elapsed_time_progress(self):
        state = SimpleNamespace(logical_step=2, current_time=2.0)
        moments = iter([0.0, 31.0])
        reporter = ProgressReporter(10000.0, 10000, 1000, clock=lambda: next(moments))
        with self.assertLogs("mf6pqc.results", level="INFO") as logs:
            reporter.report(state)
        self.assertIn("step=2/10000", logs.output[0])


class NumericalBoundaryTests(unittest.TestCase):
    def test_cached_domains_preserve_signed_components_and_match_reference(self):
        names = ["A", "B", "Charge", "C"]
        slices = tuple(slice(i * 2, (i + 1) * 2) for i in range(4))
        blocks = build_nonnegative_slices(names, slices, {"charge"})
        self.assertEqual(blocks, (slice(0, 4), slice(6, 8)))
        source = np.array([-1.0, 2.0, 0.0, -3.0, -0.1, 0.2, -1.0, 3.0])
        expected = source.copy()
        expected[[0, 2, 3, 6]] = 1e-20
        enforce_component_domains(source, names, slices, nonnegative_slices=blocks)
        np.testing.assert_array_equal(source, expected)

    def test_porosity_limit_is_diagnosed_and_strict_mode_preserves_old_state(self):
        sim, state, _ = make_case()
        sim.selected_output[:] = 100.0
        sim.porosity[-1] = 5e-5
        sim.porosity_clipping = {}
        sim.fail_on_porosity_clipping = True
        original = sim.porosity.copy()
        with self.assertRaises(PropertyUpdateError):
            update_medium_properties(sim, state.current_k11, 0)
        np.testing.assert_array_equal(sim.porosity, original)
        sim.fail_on_porosity_clipping = False
        with self.assertLogs("mf6pqc.feedback", level="WARNING"):
            update_medium_properties(sim, state.current_k11, 0)
        self.assertEqual(sim.porosity_clipping["cells"], 2)
        self.assertEqual(sim.porosity_clipping["updates"], 1)
        np.testing.assert_array_equal(sim.porosity, [1e-4, 1e-4, original[2]])

    def test_swig_filter_keeps_unrelated_deprecations_visible(self):
        fake = SimpleNamespace(PhreeqcRM=lambda *_: object())
        with (
            warnings.catch_warnings(record=True) as caught,
            patch.dict("sys.modules", {"phreeqcrm": fake}),
        ):
            warnings.simplefilter("always")
            NativeBackendFactory().create_phreeqcrm(1, 1)
            for name in ("SwigPyPacked", "SwigPyObject", "swigvarlink"):
                warnings.warn(
                    f"builtin type {name} has no __module__ attribute",
                    DeprecationWarning,
                    stacklevel=1,
                )
            warnings.warn("unrelated deprecation", DeprecationWarning, stacklevel=1)
        self.assertEqual([str(item.message) for item in caught], ["unrelated deprecation"])

    def test_preallocated_chemistry_matches_pipe_shared_memory_and_native_adapter(self):
        initial = np.arange(15, dtype=float)
        for shared in (False, True, None):
            with self.subTest(shared=shared):
                backend = (
                    Reactor(5)
                    if shared is None
                    else _ProcessPhreeqcRM(5, 2, shared_memory=shared, constructor=Reactor)
                )
                if shared is not None:
                    self.addCleanup(backend.close)
                checked = CheckedPhreeqcRM(backend)
                checked.SetConcentrations(initial)
                checked.GetSelectedOutput()
                reacted = np.empty_like(initial)
                selected = np.empty((2, 5))
                for dt in (0.25, 0.5):
                    checked.advance_into(initial, 0.0, dt, reacted, selected)
                    self.assertIs(checked.GetConcentrations(), reacted)
                    np.testing.assert_array_equal(reacted, initial + dt)
                    np.testing.assert_array_equal(
                        checked.GetSelectedOutput().reshape(2, 5),
                        (initial.reshape(3, 5) + dt)[[0, 2]],
                    )
                snapshot = selected.copy()
                checked.commit_porosity(np.full(5, 0.5))
                np.testing.assert_array_equal(checked.GetConcentrations(), initial + 0.5)
                np.testing.assert_array_equal(selected, snapshot)


class ConfigurationRegressionTests(unittest.TestCase):
    def test_counts_and_step_schedules_never_truncate(self):
        for field in (
            "nxyz",
            "nthreads",
            "save_interval",
            "progress_interval",
            "sia_max_iterations",
        ):
            for value in (True, np.bool_(True), 1.5, np.nan, np.inf, 0, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ConfigurationError):
                    MF6PQC(**{field: value})
        for field in ("save_steps", "reaction_steps"):
            for value in ([1.5], [True], [np.nan], [0], [], "12"):
                with self.subTest(field=field, value=value), self.assertRaises(ConfigurationError):
                    MF6PQC(**{field: value})

    def test_tolerances_and_masks(self):
        for options in ({"sia_rtol": np.nan}, {"sia_atol": np.inf}, {"sia_rtol": 0, "sia_atol": 0}):
            with self.subTest(options=options), self.assertRaises(ConfigurationError):
                MF6PQC(**options)
        for field in ("print_chemistry_mask", "porosity_update_mask"):
            for value in (np.nan, -1, 0.2, 2):
                with self.subTest(field=field, value=value), self.assertRaises(ConfigurationError):
                    MF6PQC(nxyz=1, **{field: value})

    def test_indices_check_both_int32_bounds_before_cast(self):
        for value in (-4294967296, [-4294967296], 2147483648, [2147483648], True, [True], 1.1):
            with (
                self.subTest(value=value),
                self.assertRaises((ValueError, OverflowError, TypeError)),
            ):
                create_ic_array_from_map(1, {"solution": value})
        np.testing.assert_array_equal(create_ic_array_from_map(1, {"solution": -1}), [-1] * 7)

    def test_case_name_cannot_be_a_path(self):
        for name in ("../other", r"a\b", ".", "x:y"):
            with self.subTest(name=name), self.assertRaises(ConfigurationError):
                MF6PQC(case_name=name)


class BackendRegressionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "db").touch()
        (self.root / "input").touch()
        self.factory = FakeFactory()

    def simulator(self, **options):
        sim = MF6PQC(
            nxyz=2,
            db_path=self.root / "db",
            pqi_path=self.root / "input",
            output_dir=self.root / "out",
            backend_factory=self.factory,
            **options,
        )
        self.addCleanup(sim.finalize)
        return sim

    def test_status_failure_raises_but_signed_getters_survive(self):
        checked = CheckedPhreeqcRM(
            SimpleNamespace(
                RunCells=lambda: -3, GetErrorString=lambda: "bad chemistry", GetValue=lambda: -1
            )
        )
        with self.assertRaisesRegex(BackendError, "RunCells.*bad chemistry"):
            checked.RunCells()
        self.assertEqual(checked.GetValue(), -1)

    def test_failed_instances_cannot_reenter_native_solvers(self):
        sim = self.simulator()
        sim.setup({"solution": 0})

        def fail(_):
            raise RuntimeError("native solve failed")

        with self.assertRaisesRegex(RuntimeError, "native solve failed"):
            sim._run_coupling(fail)
        self.assertEqual(self.factory.chemistry.closed, 1)
        self.assertEqual(self.factory.chemistry.broken, 1)
        with self.assertRaisesRegex(CouplingError, "finalized"):
            sim.setup({"solution": 0})
        with self.assertRaisesRegex(CouplingError, "finalized"):
            sim.run()
        sim.finalize()
        self.assertIsNone(sim.phreeqc_rm)
        self.assertIsNone(sim.modflow_api)
        self.assertEqual(self.factory.chemistry.closed, 1)

    def test_hook_failure_closes_backends_and_clears_run_state(self):
        sim = self.simulator()
        sim.setup({"solution": 0})

        def fail(sim, state):
            raise RuntimeError("audit failed")

        with self.assertRaisesRegex(RuntimeError, "audit failed"):
            sim._run_coupling(
                lambda instance: instance._coupling_hooks.on_step(instance, None),
                CouplingMethod.SNIA,
                hooks=CouplingHooks(on_step=fail),
            )
        self.assertFalse(sim._run_active)
        self.assertFalse(sim._run_completed)
        self.assertIsNone(sim._coupling_hooks)
        self.assertIsNone(sim.phreeqc_rm)

    def test_phase_hooks_are_rejected_for_unsupported_algorithms(self):
        sim = self.simulator()
        with self.assertRaisesRegex(ConfigurationError, "require SNIA"):
            sim.run("Strang", hooks=CouplingHooks(on_transport=lambda *_: None))
        self.assertFalse(sim._run_active)

    def test_worker_cleanup_survives_file_close_failure(self):
        sim = self.simulator()

        def fail():
            raise RuntimeError("close failed")

        self.factory.chemistry.CloseFiles = fail
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            sim.finalize()
        self.assertEqual(self.factory.chemistry.broken, 1)
        self.assertTrue(any("CloseFiles" in str(item.message) for item in caught))

    def test_density_never_overwrites_unrelated_selected_output(self):
        sim = self.simulator(if_update_density=True, use_phreeqc_calculated_density=True)
        self.factory.chemistry.GetDensityCalculated = lambda: np.array([1.1, 1.2])
        sim.setup({"solution": 0})
        np.testing.assert_array_equal(sim.selected_output, [[0.25, 0.25]])
        update_selected_output(sim)
        np.testing.assert_array_equal(sim.selected_output, [[0.25, 0.25]])
        self.factory.chemistry.GetSelectedOutput = lambda: np.ones(4)
        with self.assertRaisesRegex(BackendError, "size changed"):
            update_selected_output(sim)

    def test_time_units_and_ats_fail_before_native_loading(self):
        (self.root / "mfsim.nam").write_text('TDIS6 "model time.tdis"\n', encoding="utf-8")
        tdis = self.root / "model time.tdis"
        for units in ("SECONDS", "MINUTES", "HOURS", "YEARS", "UNKNOWN"):
            tdis.write_text(f"TIME_UNITS {units}\n", encoding="utf-8")
            with self.subTest(units=units), self.assertRaisesRegex(ConfigurationError, "DAYS"):
                validate_modflow_workspace(self.root)
        tdis.write_text("TIME_UNITS DAYS\nATS6 FILEIN model.ats\n", encoding="utf-8")
        with self.assertRaisesRegex(ConfigurationError, "ATS"):
            validate_modflow_workspace(self.root)
        tdis.write_text("TIME_UNITS DAYS # comment\n", encoding="utf-8")
        validate_modflow_workspace(self.root)
