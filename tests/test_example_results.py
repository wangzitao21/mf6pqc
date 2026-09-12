from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mf6pqc.output_processing import save_results

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
if not (EXAMPLES / "example_utils.py").is_file():
    raise unittest.SkipTest("Example sources are not included in this distribution")

spec = importlib.util.spec_from_file_location(
    "example_result_helpers", EXAMPLES / "example_utils.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


class ExampleResultTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write_results(self, values, times):
        save_results(
            self.directory / "output",
            "test",
            ["Na", "Ca", "T"],
            values,
            [],
            [],
            [],
            False,
            False,
            result_times=times,
        )

    def test_loaded_results_and_views_allow_replacing_saved_files(self):
        original = np.arange(24.0).reshape(2, 3, 4)
        self.write_results(original, [0.0, 1.0])
        values, headings, times = helpers.load_results(self.directory / "output")
        last_cell = values[:, :, -1]
        replacement = np.arange(45.0).reshape(3, 3, 5) + 100.0
        try:
            self.write_results(replacement, [0.0, 0.5, 1.0])
            updated, updated_headings, updated_times = helpers.load_results(
                self.directory / "output"
            )
            self.assertNotIsInstance(values, np.memmap)
            np.testing.assert_array_equal(values, original)
            np.testing.assert_array_equal(last_cell, original[:, :, -1])
            np.testing.assert_array_equal(times, [0.0, 1.0])
            np.testing.assert_array_equal(updated, replacement)
            np.testing.assert_array_equal(updated_times, [0.0, 0.5, 1.0])
            self.assertEqual(headings, updated_headings)
        finally:
            if isinstance(values, np.memmap):
                values._mmap.close()

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "Plot checks require matplotlib")
    def test_pht3d06_plot_reads_reference_and_keeps_simulation_metadata(self):
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self.addCleanup(plt.close, "all")
        times = np.array([0.0, 0.1, 0.2])
        values = np.arange(18.0).reshape(3, 3, 2) / 10000.0
        self.write_results(values, times)
        input_dir = self.directory / "input_data"
        input_dir.mkdir()
        reference = np.zeros(
            (), dtype=[(name, float, (3,)) for name in ("time_days", "Ca", "Na", "T")]
        )
        reference["time_days"] = [0.0, 0.075, 0.2]
        for index, name in enumerate(("Ca", "Na", "T"), start=1):
            reference[name] = np.array([1.0, 2.0, 3.0]) * index / 10000.0
        np.save(input_dir / "PHT3D_06_results.npy", reference)
        observations = np.zeros(
            (), dtype=[(name, float, (2,)) for name in ("time_minutes", "Ca", "Na", "T")]
        )
        observations["time_minutes"] = [10.0, 20.0]
        np.save(input_dir / "observations.npy", observations)
        notebook_path = EXAMPLES / "ex006_PHT3D_06" / "plot.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        namespace = {
            "INPUT_DIR": input_dir,
            "OUTPUT_DIR": self.directory / "output",
            "load_results": helpers.load_results,
        }
        with patch.object(plt, "show"):
            for cell in notebook["cells"][1:]:
                exec(compile("".join(cell["source"]), str(notebook_path), "exec"), namespace)
        np.testing.assert_array_equal(namespace["result_times"], times)
        self.assertEqual(namespace["headings"], ["Na", "Ca", "T"])
        figure = namespace["fig"]
        figure.canvas.draw()
        self.assertEqual(len(figure.axes), 3)
        for axis, name, index in zip(figure.axes, ("Ca", "T", "Na"), (1, 2, 0), strict=True):
            self.assertEqual(len(axis.lines), 3)
            np.testing.assert_array_equal(
                axis.lines[0].get_xdata(), reference["time_days"] * 24 * 60
            )
            np.testing.assert_allclose(
                axis.lines[0].get_ydata(),
                np.interp(reference["time_days"], times, values[:, index, -1] * 1000),
            )
            np.testing.assert_array_equal(axis.lines[1].get_ydata(), reference[name] * 1000)
            self.assertEqual(axis.get_ylabel(), f"{name} (mmol/L)")


if __name__ == "__main__":
    unittest.main()
