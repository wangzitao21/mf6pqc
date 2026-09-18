"""Shared executable-example contracts, independent of individual benchmarks."""

import ast
import tempfile
import unittest
from pathlib import Path

from scripts.check_examples import (
    EXAMPLES,
    REQUIRED_FILES,
    check_imports,
    check_syntax,
    discover_cases,
)


class ExampleTests(unittest.TestCase):
    def test_all_cases_have_valid_sources_and_safe_imports(self):
        cases = discover_cases()
        self.assertFalse(list(EXAMPLES.glob("*.py")))
        for case in cases:
            with self.subTest(case=case.name):
                check_syntax(case)
                source = ast.parse((case / "run.py").read_text(encoding="utf-8"))
                main = next(
                    n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main"
                )
                self.assertFalse(main.args.args or main.args.kwonlyargs)
                imports = {
                    a.name for n in ast.walk(source) if isinstance(n, ast.Import) for a in n.names
                }
                self.assertFalse(imports & {"argparse", "subprocess"})
        check_imports(cases)

    def test_missing_or_extra_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / "ex001_Test"
            (case / "input_data").mkdir(parents=True)
            for name in REQUIRED_FILES:
                (case / name).touch()
            self.assertEqual(discover_cases(root), [case])
            (case / "helper.py").touch()
            with self.assertRaisesRegex(AssertionError, "unexpected"):
                discover_cases(root)
            (case / "helper.py").unlink()
            (case / "run.py").unlink()
            with self.assertRaisesRegex(AssertionError, "missing"):
                discover_cases(root)
