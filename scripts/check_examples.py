"""Check example layouts, notebook syntax, and imports without running solvers."""

from __future__ import annotations

import ast
import importlib
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REQUIRED_FILES = {"run.py", "modflow_model.py", "plot.ipynb"}
ALLOWED_ENTRIES = REQUIRED_FILES | {
    "input_data",
    "output",
    "simulation",
    "__pycache__",
    ".ipynb_checkpoints",
}
sys.path.insert(0, str(ROOT))


def discover_cases(examples: Path = EXAMPLES) -> list[Path]:
    cases = sorted(path for path in examples.glob("ex[0-9]*") if path.is_dir())
    if not cases:
        raise AssertionError(f"No cases found in {examples}")
    for case in cases:
        unexpected = {path.name for path in case.iterdir()} - ALLOWED_ENTRIES
        if unexpected:
            raise AssertionError(f"{case.name}: unexpected entries {sorted(unexpected)}")
        for name in REQUIRED_FILES:
            if not (case / name).is_file():
                raise AssertionError(f"{case.name}: missing {name}")
        if not (case / "input_data").is_dir():
            raise AssertionError(f"{case.name}: missing input_data")
    return cases


def check_syntax(case: Path) -> None:
    import nbformat
    from IPython.core.inputtransformer2 import TransformerManager

    for path in case.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    notebook = nbformat.read(case / "plot.ipynb", as_version=4)
    nbformat.validate(notebook)
    transform = TransformerManager().transform_cell
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(transform(cell.source), filename=f"{case.name}/plot.ipynb:{index}")
            if any(output.output_type == "error" for output in cell.outputs):
                raise AssertionError(f"{case.name}: saved notebook execution error")


def check_imports(cases: list[Path]) -> None:
    import flopy

    from mf6pqc import MF6PQC

    with ExitStack() as stack:
        for owner, method in ((MF6PQC, "_initialize"), (flopy.mf6.MFSimulation, "__init__")):
            stack.enter_context(
                patch.object(
                    owner, method, side_effect=AssertionError("Solver created during import")
                )
            )
        for case in cases:
            for name in ("modflow_model", "run"):
                importlib.import_module(f"examples.{case.name}.{name}")


def main() -> None:
    cases = discover_cases()
    for case in cases:
        check_syntax(case)
    check_imports(cases)
    print(f"{len(cases)} examples: layouts, syntax, and imports passed.")


if __name__ == "__main__":
    main()
