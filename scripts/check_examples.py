"""Check example layouts and safe imports, or explicitly run selected cases.

Long-running benchmarks are excluded from this native regression command.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import runpy
import subprocess
import sys
import time
from contextlib import ExitStack
from importlib.abc import MetaPathFinder
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
PHT3D_SMOKE_CASES = tuple(f"ex{i:03}_PHT3D_{i:02}" for i in range(1, 11))
NATIVE_CASES = {
    name: [("run.py", [])]
    for name in [
        *PHT3D_SMOKE_CASES,
        "ex999_Thermal_ReactiveColumn1D",
        "ex019_Splitting_KineticDecay1D",
        "ex020_Splitting_RedoxFront2D",
    ]
}
REQUIRED_FILES = {"modflow_model.py", "run.py", "plot.ipynb"}
OPTIONAL_FILES = {
    "README.md",
    "config.py",
    "comparison.py",
    "chemistry.py",
    "simulation.py",
    "analysis.py",
}
RUNTIME_DIRECTORIES = {"output", "simulation"}
CACHE_DIRECTORIES = {"__pycache__", ".ipynb_checkpoints"}
SHARED_FILES = {"example_utils.py", "README.md"}
ALLOWED_ENTRIES = (
    REQUIRED_FILES | OPTIONAL_FILES | RUNTIME_DIRECTORIES | CACHE_DIRECTORIES | {"input_data"}
)


def discover_cases(examples: Path) -> list[Path]:
    """Validate source layouts without requiring generated runtime directories."""
    cases = []
    for entry in sorted(examples.iterdir()):
        if entry.is_file() and entry.name in SHARED_FILES:
            continue
        if entry.is_dir() and entry.name in CACHE_DIRECTORIES:
            continue
        if not entry.is_dir() or not re.fullmatch(r"ex\d{3}_[A-Za-z0-9_]+", entry.name):
            raise AssertionError(f"Unexpected entry in examples: {entry.name}")
        unexpected = {path.name for path in entry.iterdir()} - ALLOWED_ENTRIES
        if unexpected:
            raise AssertionError(f"{entry.name}: unexpected entries: {sorted(unexpected)}")
        for name in REQUIRED_FILES:
            if not (entry / name).is_file():
                raise AssertionError(f"{entry.name}: missing {name}")
        if not (entry / "input_data").is_dir():
            raise AssertionError(f"{entry.name}: missing input_data directory")
        for name in RUNTIME_DIRECTORIES | CACHE_DIRECTORIES:
            if (entry / name).exists() and not (entry / name).is_dir():
                raise AssertionError(f"{entry.name}: {name} must be a directory")
        for name in OPTIONAL_FILES:
            if (entry / name).exists() and not (entry / name).is_file():
                raise AssertionError(f"{entry.name}: {name} must be a file")
        cases.append(entry)
    if not cases:
        raise AssertionError("No example cases found")
    if not (examples / "example_utils.py").is_file():
        raise AssertionError("Missing shared example_utils.py")
    return cases


def _import_without_solvers(scripts: list[Path]) -> None:
    """Import modules with backend creation, model runs and child processes blocked."""
    import flopy

    from mf6pqc.backends import NativeBackendFactory

    # NumPy/SciPy query platform information during import. On Windows,
    # a cold uname cache can launch the harmless system command "ver".
    # Resolve it (and the lazy processor field) before blocking every
    # child process launched by the example modules.
    platform.processor()

    attempted = []

    def forbidden(*args, **kwargs):
        message = "Solver creation or execution attempted during import"
        attempted.append(message)
        raise AssertionError(message)

    class NativeImportGuard(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in {"phreeqcrm", "modflowapi"}:
                forbidden()
            return None

    guard = NativeImportGuard()
    with ExitStack() as stack:
        for owner, attribute in (
            (NativeBackendFactory, "create_phreeqcrm"),
            (NativeBackendFactory, "create_modflow_api"),
            (NativeBackendFactory, "load_modflow_simulation"),
            (flopy.mf6.MFSimulation, "__init__"),
            (flopy.mf6.MFSimulation, "write_simulation"),
            (flopy.mf6.MFSimulation, "run_simulation"),
            (flopy, "run_model"),
            (flopy.mbase, "run_model"),
            (subprocess, "Popen"),
            (os, "system"),
        ):
            stack.enter_context(patch.object(owner, attribute, side_effect=forbidden))
        # Normal check subprocesses have neither native package imported. Also
        # guard existing module objects for callers using this helper directly.
        for module_name, attribute in (("phreeqcrm", "PhreeqcRM"), ("modflowapi", "ModflowApi")):
            if module_name in sys.modules:
                stack.enter_context(
                    patch.object(sys.modules[module_name], attribute, side_effect=forbidden)
                )
        sys.meta_path.insert(0, guard)
        stack.callback(sys.meta_path.remove, guard)
        for script in scripts:
            runpy.run_path(str(script), run_name="__import_check__")
        if attempted:
            raise AssertionError(attempted[0])


def check_imports(scripts: list[Path], *, cwd: Path) -> None:
    """Give each case a fresh module cache without invoking its main entry point."""
    code = """import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / 'examples'))
from scripts.check_examples import _import_without_solvers
_import_without_solvers([Path(filename) for filename in sys.argv[2:]])
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), *map(str, scripts)],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=45,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONIOENCODING": "utf-8"},
    )
    if completed.returncode:
        raise AssertionError(f"{cwd.name}: import check failed\n{completed.stderr}")


def check_syntax(paths: list[Path]) -> None:
    """Parse Python files and transformed notebook code without executing cells."""
    from IPython.core.inputtransformer2 import TransformerManager

    transformer = TransformerManager()
    for path in paths:
        if path.suffix == ".py":
            ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        elif path.suffix == ".ipynb":
            notebook = json.loads(path.read_text(encoding="utf-8-sig"))
            for index, cell in enumerate(notebook["cells"], start=1):
                if cell["cell_type"] != "code":
                    continue
                if any(o.get("output_type") == "error" for o in cell.get("outputs", [])):
                    raise AssertionError(f"{path}: cell {index} includes an execution error")
                ast.parse(
                    transformer.transform_cell("".join(cell["source"])),
                    filename=f"{path}:cell {index}",
                )


def check_static(*, allow_notebook_outputs: bool = False) -> list[dict]:
    """Parse all source code and import cases; saved notebook figures are allowed."""
    cases = discover_cases(EXAMPLES)
    shared = sorted(EXAMPLES.glob("*.py"))
    check_syntax(shared)
    check_imports(shared, cwd=EXAMPLES)
    reports = []
    for case in cases:
        source_paths = sorted(
            path
            for path in case.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".ipynb"}
            and path.relative_to(case).parts[0] not in RUNTIME_DIRECTORIES
            and not CACHE_DIRECTORIES.intersection(path.relative_to(case).parts)
        )
        check_syntax(source_paths)
        check_imports(sorted(case.glob("*.py")), cwd=case)
        reports.append(
            {"case": case.name, "static_imports": "passed", "simulation_executed": False}
        )
        print(f"{case.name}: layout, Python/notebook syntax, guarded imports passed", flush=True)
    return reports


def check_native(names: list[str], output_root: Path, timeout: float) -> list[dict]:
    """Run only explicitly named catalogue cases in an isolated result directory."""
    if not names or set(names) - NATIVE_CASES.keys():
        raise ValueError("Select at least one supported native case explicitly")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
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
