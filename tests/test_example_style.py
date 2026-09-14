"""Regression checks for example discovery and imports, without native solvers."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts import check_examples

HAS_EXAMPLE_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None for name in ("flopy", "IPython")
)


class ExampleLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.examples = Path(self.temporary.name)
        (self.examples / "example_utils.py").write_text("", encoding="utf-8")
        self.case = self.examples / "ex001_Test"
        self.case.mkdir()
        (self.case / "input_data").mkdir()
        for name in check_examples.REQUIRED_FILES:
            (self.case / name).write_text("", encoding="utf-8")

    def test_case_has_three_source_files_and_three_directory_names(self):
        self.assertEqual(
            check_examples.ALLOWED_ENTRIES,
            {"run.py", "modflow_model.py", "plot.ipynb", "input_data", "output", "simulation"},
        )
        self.assertEqual(check_examples.discover_cases(self.examples), [self.case])
        for name in ("output", "simulation"):
            (self.case / name).mkdir()
        self.assertEqual(check_examples.discover_cases(self.examples), [self.case])

    def test_unexpected_source_and_invalid_runtime_entry_are_rejected(self):
        unexpected = self.case / "old_run.py"
        unexpected.touch()
        with self.assertRaisesRegex(AssertionError, "unexpected entries"):
            check_examples.discover_cases(self.examples)
        unexpected.unlink()
        (self.case / "output").touch()
        with self.assertRaisesRegex(AssertionError, "must be a directory"):
            check_examples.discover_cases(self.examples)

    def test_extra_case_files_and_directories_are_rejected(self):
        for name in ("config.py", "analysis.py", "output_original_20260913_174349", "__pycache__"):
            extra = self.case / name
            extra.touch()
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(AssertionError, "unexpected entries"),
            ):
                check_examples.discover_cases(self.examples)
            extra.unlink()

    def test_native_catalogue_refers_to_current_source_directories(self):
        self.assertEqual(len(check_examples.PHT3D_SMOKE_CASES), 10)
        for name, commands in check_examples.NATIVE_CASES.items():
            for script, _ in commands:
                with self.subTest(case=name):
                    self.assertTrue((check_examples.EXAMPLES / name / script).is_file())

    def test_no_native_selection_is_implicit(self):
        with (
            patch.object(sys, "argv", ["check_examples.py"]),
            patch.object(check_examples, "check_static") as static,
            patch.object(check_examples, "check_native") as native,
        ):
            check_examples.main()
        static.assert_called_once()
        native.assert_not_called()
        for names in ([], ["../unlisted"]):
            with self.subTest(names=names), self.assertRaises(ValueError):
                check_examples.check_native(names, self.examples / "unused", 1)
        self.assertFalse((self.examples / "unused").exists())

    def test_successful_process_without_results_is_not_accepted(self):
        with self.assertRaisesRegex(AssertionError, "No completed result"):
            check_examples.validate_native_results(self.case)

    def test_native_selection_runs_only_named_cases_with_relocated_outputs(self):
        selected = check_examples.PHT3D_SMOKE_CASES[0]
        output = self.examples / "native"
        with (
            patch.object(
                check_examples.subprocess, "run", return_value=SimpleNamespace(returncode=0)
            ) as run,
            patch.object(
                check_examples, "validate_native_results", return_value={"result_sets": 1}
            ) as validate,
        ):
            reports = check_examples.check_native([selected], output, 10)
        validate.assert_called_once_with(output / selected)
        run.assert_called_once()
        self.assertEqual(
            run.call_args.args[0][1], str(check_examples.EXAMPLES / selected / "run.py")
        )
        self.assertEqual(run.call_args.kwargs["env"]["MF6PQC_RUN_ROOT"], str(output))
        self.assertEqual([report["case"] for report in reports], [selected])


class NativeResultValidationTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.case = Path(self.temporary.name)
        self.output = self.case / "output"
        self.output.mkdir()
        self.manifest = {
            "run": {"completed": True, "nxyz": 3},
            "result_shape": [2, 2, 3],
            "files": ["results.npy", "results_times.npy", "results_headings.txt"],
        }
        self.write_manifest()
        np.save(self.output / "results.npy", np.ones((2, 2, 3)))
        np.save(self.output / "results_times.npy", [0.0, 1.0])
        (self.output / "results_headings.txt").write_text("A\nB\n", encoding="utf-8")

    def write_manifest(self):
        (self.output / "results_manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8"
        )

    def test_valid_result_set(self):
        self.assertEqual(
            check_examples.validate_native_results(self.case),
            {"result_sets": 1, "arrays_checked": 2},
        )

    def test_incomplete_or_nonconvergent_run_is_rejected(self):
        for run in (
            {"completed": False},
            {"completed": True, "modflow_convergence_failures": [1]},
            {"completed": True, "sia_convergence_failures": [1]},
        ):
            with self.subTest(run=run):
                self.manifest["run"] = run
                self.write_manifest()
                with self.assertRaises(AssertionError):
                    check_examples.validate_native_results(self.case)

    def test_missing_and_corrupt_saved_arrays_are_rejected(self):
        import numpy as np

        np.save(self.output / "results.npy", np.full((2, 2, 3), np.nan))
        with self.assertRaisesRegex(AssertionError, "Non-finite"):
            check_examples.validate_native_results(self.case)
        np.save(self.output / "results.npy", np.ones((2, 3, 2)))
        with self.assertRaisesRegex(AssertionError, "shape"):
            check_examples.validate_native_results(self.case)
        (self.output / "results.npy").unlink()
        with self.assertRaisesRegex(AssertionError, "Missing output"):
            check_examples.validate_native_results(self.case)

    def test_invalid_times_and_physical_properties_are_rejected(self):
        import numpy as np

        for times in ([0.0, 0.0], [1.0, 0.0], [0.0, np.nan]):
            np.save(self.output / "results_times.npy", times)
            with self.subTest(times=times), self.assertRaises(AssertionError):
                check_examples.validate_native_results(self.case)
        np.save(self.output / "results_times.npy", [0.0, 1.0])
        for name, values in (
            ("results_porosity.npy", [0.0, 0.3]),
            ("results_porosity.npy", [1.1, 0.3]),
            ("results_K.npy", [-1.0, 1.0]),
        ):
            np.save(self.output / name, values)
            with self.subTest(name=name, values=values), self.assertRaises(AssertionError):
                check_examples.validate_native_results(self.case)
            (self.output / name).unlink()


@unittest.skipUnless(HAS_EXAMPLE_DEPENDENCIES, "Requires the optional example dependencies")
class ExampleImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.script = self.directory / "module.py"

    def test_native_creation_flopy_runs_and_child_processes_are_blocked(self):
        sources = (
            "from mf6pqc.backends import NativeBackendFactory\nNativeBackendFactory().create_phreeqcrm(1, 1)",
            "from mf6pqc import ProcessBackendFactory\nProcessBackendFactory().create_phreeqcrm(1, 1)",
            "from mf6pqc.backends import NativeBackendFactory\nNativeBackendFactory().create_modflow_api('missing', '.')",
            "import phreeqcrm\nphreeqcrm.PhreeqcRM(1, 1)",
            "import modflowapi\nmodflowapi.ModflowApi('missing')",
            "import flopy\nflopy.mf6.MFSimulation.run_simulation(None)",
            "import flopy\nflopy.run_model('missing', 'missing')",
            "import subprocess\nsubprocess.run(['missing'])",
        )
        for source in sources:
            self.script.write_text(source, encoding="utf-8")
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(AssertionError, "during import"),
            ):
                check_examples._import_without_solvers([self.script])

    @unittest.skipUnless(sys.platform == "win32", "Windows platform probe regression")
    def test_windows_cold_platform_cache_is_warmed_without_allowing_example_processes(self):
        # Load the guard's own dependencies first so their imports cannot warm
        # the cache and conceal the cold-cache path being exercised below.
        check_examples._import_without_solvers([])
        self.script.write_text(
            "import platform\nassert platform.machine()\nplatform.processor()\n",
            encoding="utf-8",
        )
        with (
            patch.object(platform, "_uname_cache", None),
            patch.object(platform, "win32_ver", wraps=platform.win32_ver) as version_probe,
        ):
            check_examples._import_without_solvers([self.script])
            version_probe.assert_called_once()
            self.script.write_text(
                "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', 'pass'])\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(AssertionError, "during import"):
                check_examples._import_without_solvers([self.script])
            version_probe.assert_called_once()

    def test_caught_solver_attempt_still_fails_check(self):
        self.script.write_text(
            "import flopy\ntry:\n    flopy.mf6.MFSimulation()\nexcept AssertionError:\n    pass\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(AssertionError, "during import"):
            check_examples._import_without_solvers([self.script])

    def test_main_is_not_invoked_and_dataclasses_can_import(self):
        self.script.write_text(
            "from dataclasses import dataclass\n"
            "@dataclass\nclass Config:\n    value: int = 1\n"
            "if __name__ == '__main__':\n    raise RuntimeError('main executed')\n",
            encoding="utf-8",
        )
        check_examples.check_imports([self.script], cwd=self.directory)

    def test_python_and_notebook_syntax_are_checked_without_executing_code(self):
        self.script.write_text("raise RuntimeError('must not execute')\n", encoding="utf-8")
        notebook = self.directory / "plot.ipynb"
        cells = [
            {
                "cell_type": "code",
                "source": ["%matplotlib inline\n", "raise RuntimeError()"],
                "outputs": [],
            }
        ]
        notebook.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        check_examples.check_syntax([self.script, notebook])
        self.script.write_text("def broken(\n", encoding="utf-8")
        with self.assertRaises(SyntaxError):
            check_examples.check_syntax([self.script])
        cells[0]["source"] = ["def broken(\n"]
        notebook.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        with self.assertRaises(SyntaxError):
            check_examples.check_syntax([notebook])
        cells[0]["source"] = ["x = 1"]
        cells[0]["outputs"] = [{"output_type": "error", "ename": "RuntimeError"}]
        notebook.write_text(json.dumps({"cells": cells}), encoding="utf-8")
        with self.assertRaisesRegex(AssertionError, "execution error"):
            check_examples.check_syntax([notebook])


if __name__ == "__main__":
    unittest.main()
