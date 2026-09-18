"""Verify runtime contents and source completeness in built distributions."""

import argparse
import ast
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


def is_documentation(name: str) -> bool:
    path = Path(name)
    return (
        path.suffix.lower() in {".md", ".rst", ".pdf", ".html", ".htm", ".doc", ".docx", ".rtf"}
        or path.name.upper().startswith(("README", "CHANGELOG", "CONTRIBUTING", "CITATION"))
        or bool({"docs", ".github"}.intersection(path.parts))
    )


def check_distribution(directory: Path) -> None:
    wheels, sources = list(directory.glob("*.whl")), list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise AssertionError("Expected exactly one wheel and one source archive")
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        if any(is_documentation(name) for name in names):
            raise AssertionError("Wheel contains documentation")
        metadata = BytesParser().parsebytes(
            wheel.read(next(n for n in names if n.endswith(".dist-info/METADATA")))
        )
        package, version = metadata["Name"], metadata["Version"]
        tree = ast.parse(wheel.read(f"{package}/_version.py"))
        runtime_version = next(
            ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
        )
        if version != runtime_version:
            raise AssertionError("Wheel and runtime versions differ")
        if f"{package}/py.typed" not in names:
            raise AssertionError("Missing type marker")
        if any(not (n.startswith(f"{package}/") or ".dist-info/" in n) for n in names):
            raise AssertionError("Wheel contains files outside the runtime package")
        runtime_files = {n for n in names if n.startswith(f"{package}/")}
    with tarfile.open(sources[0]) as source:
        members = {m.name.partition("/")[2]: m for m in source.getmembers() if m.isfile()}
        metadata = BytesParser().parsebytes(source.extractfile(members["PKG-INFO"]).read())
        if (metadata["Name"], metadata["Version"]) != (package, version):
            raise AssertionError("Wheel and source metadata differ")
        required = runtime_files | {"LICENSE", "pyproject.toml"}
        root = Path(__file__).resolve().parents[1]
        for folder in ("examples", "scripts", "tests"):
            for path in (root / folder).rglob("*"):
                if path.is_file() and not is_documentation(path.relative_to(root).as_posix()) and not {
                    "output",
                    "simulation",
                    "__pycache__",
                    ".ipynb_checkpoints",
                }.intersection(path.relative_to(root).parts):
                    required.add(path.relative_to(root).as_posix())
        if missing := required - members.keys():
            raise AssertionError(f"Source archive is missing {sorted(missing)}")
        for name in members:
            if is_documentation(name):
                raise AssertionError(f"Source archive contains documentation: {name}")
            parts = Path(name).parts
            if {"__pycache__", ".ipynb_checkpoints", "bin", "cases", "tmp"}.intersection(parts):
                raise AssertionError(f"Unwanted source file: {name}")
            if (
                parts[0] == "examples"
                and len(parts) > 3
                and parts[2] in {"output", "simulation"}
                and parts[-1] != ".gitkeep"
            ):
                raise AssertionError(f"Generated example output: {name}")
    print(f"{package} {version}: wheel and source archive passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    check_distribution(parser.parse_args().directory)
