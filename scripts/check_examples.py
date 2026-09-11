"""Check self-contained example layouts and safe imports, or run selected cases.

Long-running benchmarks are excluded from this native regression command.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
NATIVE_CASES = {
    name: [("run.py", [])]
    for name in [
        *(f"PHT3D_E{i:02}" for i in range(1, 11)),
        "GWE_VSC_Reactive",
        "Splitting_KineticDecay",
        "Splitting_RedoxFront2D",
    ]
}
ALLOWED_FILES = {"modflow_model.py", "run.py", "plot.ipynb"}
ALLOWED_ENTRIES = ALLOWED_FILES | {"input_data", "output", "simulation"}


def check_static(*, allow_notebook_outputs: bool = False) -> list[dict]:
    """Parse all notebooks and import each case with native creation forbidden."""
    from IPython.core.inputtransformer2 import TransformerManager

    transformer = TransformerManager()
    reports = []
    for case in sorted(EXAMPLES.iterdir()):
        if not case.is_dir() or not (case / "run.py").is_file():
            raise AssertionError(f"Unexpected entry in examples: {case.name}")
        if {p.name for p in case.iterdir()} != ALLOWED_ENTRIES:
            raise AssertionError(f"{case.name}: expected exactly six case entries")
        for name in ("input_data", "output", "simulation"):
            if not (case / name).is_dir():
                raise AssertionError(f"{case.name}: missing {name}")
        scripts = list(case.glob("*.py"))
        if {p.name for p in scripts} - ALLOWED_FILES:
            raise AssertionError(f"{case.name}: nonstandard Python filenames")
        notebook = json.loads((case / "plot.ipynb").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                    raise AssertionError(f"{case.name}: notebook includes an execution error")
                ast.parse(transformer.transform_cell("".join(cell["source"])))
        code = """import runpy, sys
sys.dont_write_bytecode = True
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
from mf6pqc.backends import NativeBackendFactory
with (
    patch.object(NativeBackendFactory, 'create_phreeqcrm', side_effect=AssertionError('chemistry solver created during import')),
    patch.object(NativeBackendFactory, 'create_modflow_api', side_effect=AssertionError('MODFLOW solver created during import')),
):
    for filename in sys.argv[2:]:
        runpy.run_path(filename, run_name='__import_check__')
"""
        # The package import is explicit; every example gets a fresh module cache.
        completed = subprocess.run(
            [sys.executable, "-c", code, str(ROOT), *map(str, scripts)],
            cwd=case,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=45,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if completed.returncode:
            raise AssertionError(f"{case.name}: import check failed\n{completed.stderr}")
        reports.append(
            {"case": case.name, "static_imports": "passed", "simulation_executed": False}
        )
        print(f"{case.name}: layout, notebook syntax, guarded imports passed", flush=True)
    return reports


def check_native(names: list[str], output_root: Path, timeout: float) -> list[dict]:
    reports = []
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MF6PQC_RUN_ROOT"] = str(output_root)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in names:
        for script, arguments in NATIVE_CASES[name]:
            start = time.perf_counter()
            log = logs / f"{name}-{Path(script).stem}.log"
            with log.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    [sys.executable, str(EXAMPLES / name / script), *arguments],
                    cwd=output_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            report = {
                "case": name,
                "script": script,
                "arguments": arguments,
                "exit_code": completed.returncode,
                "seconds": time.perf_counter() - start,
                "log": str(log),
            }
            reports.append(report)
            (output_root / "native-report.json").write_text(
                json.dumps(reports, indent=2) + "\n", encoding="utf-8"
            )
            print(
                f"{name}/{script}: exit {completed.returncode}, {report['seconds']:.2f} s",
                flush=True,
            )
            if completed.returncode:
                raise RuntimeError(f"Native check failed. See {log}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-notebook-outputs",
        action="store_true",
        help="compatibility flag; saved notebook figures are allowed",
    )
    parser.add_argument("--native", nargs="+", choices=tuple(NATIVE_CASES), metavar="CASE")
    parser.add_argument("--output-root", type=Path, default=ROOT / ".release-checks" / "native")
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="maximum seconds for each selected case"
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.native:
        check_native(args.native, args.output_root.expanduser().resolve(), args.timeout)
    else:
        check_static(allow_notebook_outputs=args.allow_notebook_outputs)


if __name__ == "__main__":
    main()
