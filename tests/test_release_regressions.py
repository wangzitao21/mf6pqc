"""Release regression tests. No native solver is loaded."""

import json
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mf6pqc import MF6PQC
from mf6pqc.backends import CheckedPhreeqcRM, validate_modflow_workspace
from mf6pqc.coupling.common import build_time_step_schedule, update_selected_output
from mf6pqc.coupling.strang import validate_strang_schedule
from mf6pqc.exceptions import BackendError, ConfigurationError, CouplingError
from mf6pqc.input_processing import create_ic_array_from_map
from mf6pqc.output_processing import save_results
from tests.test_coupling_primitives import TdisApi
from tests.test_public_api import FakeFactory


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


class NumericalRegressionTests(unittest.TestCase):
    def test_near_one_multiplier_remains_geometric(self):
        multiplier = 1.000005
        actual = build_time_step_schedule(TdisApi([100], [1000], [multiplier]))
        np.testing.assert_allclose(actual[1:] / actual[:-1], multiplier, rtol=1e-10)
        self.assertAlmostEqual(actual.sum(), 100)
        self.assertGreater(actual[-1] / actual[0], 1.004)

    def test_noninteger_native_step_count_is_rejected(self):
        with self.assertRaisesRegex(BackendError, "integer"):
            build_time_step_schedule(TdisApi([1], [1.5], [1]))

    def test_strang_rejects_invalid_halves(self):
        for value in (0, -1, np.inf, np.nan):
            with self.subTest(value=value), self.assertRaises(CouplingError):
                validate_strang_schedule([value, value])


class SerializationRegressionTests(unittest.TestCase):
    def test_invalid_metadata_preserves_existing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.npy"
            path.write_bytes(b"existing output")
            for metadata in ({"bad": np.nan}, {"bad": object()}):
                with self.subTest(metadata=metadata), self.assertRaises((ValueError, TypeError)):
                    save_results(
                        tmp,
                        "case",
                        ["A"],
                        np.ones((1, 1, 1)),
                        [],
                        [],
                        [],
                        False,
                        False,
                        metadata=metadata,
                    )
                self.assertEqual(path.read_bytes(), b"existing output")

    def test_empty_reaction_history_has_valid_diffusion_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            save_results(
                tmp, "case", ["A"], np.ones((1, 1, 2)), [], [], [], False, True, result_times=[0]
            )
            self.assertEqual(np.load(Path(tmp) / "results_diffc.npy").shape, (0, 2))
            manifest = json.loads((Path(tmp) / "results_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["time_units"], "days")

    def test_impossible_porosity_is_not_exported(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "Porosity/K"):
            save_results(tmp, "case", ["A"], np.ones((1, 1, 1)), [[1.1]], [[1]], [], True, False)
