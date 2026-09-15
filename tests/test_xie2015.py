"""Source-unit and time-grid checks for the independent Xie benchmark cases."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from ex015_Xie2015_B2 import run as b2
from ex016_Xie2015_B3 import run as b3
from ex017_Xie2015_B4 import run as b4


class XieInputTests(unittest.TestCase):
    def test_comparisons_reject_any_missing_reference_time(self):
        import importlib.util

        path = Path(__file__).resolve().parents[1] / "scripts/validate_xie_implicit.py"
        spec = importlib.util.spec_from_file_location("xie_validation", path)
        validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(validator)
        for targets in ([10, 100, 120], [10, 100, 1000], [10, 100, 300], [100, 1000, 3000]):
            data = {"times": np.array([0, *targets], dtype=float)}
            self.assertEqual(validator.comparison_years(data, targets), targets)
            for missing in targets:
                partial = {"times": data["times"][data["times"] != missing]}
                with self.assertRaisesRegex(ValueError, "all three"):
                    validator.comparison_years(partial, targets)
                self.assertEqual(
                    validator.comparison_years(partial, targets, True),
                    [v for v in targets if v != missing],
                )

    def test_schedule_bounds_and_reference_endpoints(self):
        for runner, duration, maximum, targets in (
            (b2, 1000, 0.3, [10, 100, 1000]),
            (b3, 300, 0.1, [10, 100, 300]),
            (b4, 3000, 1.0, [100, 1000, 3000]),
        ):
            periods, saves, times = runner.schedule(duration, maximum, targets)
            dt = np.diff(np.r_[0.0, times]) / 365
            self.assertLessEqual(dt.max(), maximum * (1 + 1e-10))
            self.assertTrue(np.all(dt > 0))
            self.assertLessEqual(np.max(dt[1:] / dt[:-1]), 1.12 * (1 + 1e-9))
            for target in targets:
                self.assertLess(np.min(abs(times[np.array(saves) - 1] / 365 - target)), 1e-7)
            self.assertEqual(sum(p[1] for p in periods), len(times))

    def test_initial_total_conversion_preserves_signed_charge_and_element_ratios(self):
        names = ["H", "O", "Charge", "Ca", "C", "S", "Na"]
        expected = np.array([111.0, 55.5, -0.02, 0.0001, 0.01, 0.2, 0.39562])
        solution = {
            name: {"value": value, "constraint": "free"}
            for name, value in [
                ("ca+2", 0.0001),
                ("co3-2", 0.01),
                ("so4-2", 0.2),
                ("na+1", 0.39562),
            ]
        }
        for runner in (b2, b3, b4):
            with self.subTest(case=runner.__name__):
                actual = runner.normalize_min3p_totals(expected * 1.02673957, names, solution)
                np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
                with self.assertRaisesRegex(ValueError, "total"):
                    wrong = expected.copy()
                    wrong[5] *= 2
                    runner.normalize_min3p_totals(wrong, names, solution)


if __name__ == "__main__":
    unittest.main()
