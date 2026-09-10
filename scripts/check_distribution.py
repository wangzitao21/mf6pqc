import argparse
import ast
import json
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


def check_distribution(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise AssertionError(
            "Use a clean dist directory containing exactly one wheel and one source archive"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_files = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(metadata_files) != 1:
            raise AssertionError("Wheel must have exactly one METADATA file")
        metadata = BytesParser().parsebytes(archive.read(metadata_files[0]))
        wheel_version = metadata["Version"]
        if metadata["Name"] != "mf6pqc":
            raise AssertionError("Unexpected wheel project name")
        version_tree = ast.parse(archive.read("mf6pqc/_version.py"))
        runtime_version = next(
            ast.literal_eval(node.value)
            for node in version_tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "__version__" for t in node.targets)
        )
        if runtime_version != wheel_version:
            raise AssertionError("Runtime and wheel versions differ")
        if "mf6pqc/__init__.py" not in names or "mf6pqc/py.typed" not in names:
            raise AssertionError("Missing package entry point or type marker")
        if any(not (n.startswith("mf6pqc/") or ".dist-info/" in n) for n in names):
            raise AssertionError("Wheel contains files outside the runtime package and metadata")
    with tarfile.open(sources[0]) as archive:
        names = [PurePosixPath(n).parts[1:] for n in archive.getnames()]
        members = {
            "/".join(PurePosixPath(m.name).parts[1:]): m for m in archive.getmembers() if m.isfile()
        }

        def read_member(name):
            return archive.extractfile(members[name]).read()

        metadata = BytesParser().parsebytes(read_member("PKG-INFO"))
        if metadata["Name"] != "mf6pqc" or metadata["Version"] != wheel_version:
            raise AssertionError("Wheel and source metadata differ")
        for name in members:
            if name.endswith(".ipynb"):
                notebook = json.loads(read_member(name))
                if any(
                    c.get("outputs") or c.get("execution_count") is not None
                    for c in notebook["cells"]
                    if c["cell_type"] == "code"
                ):
                    raise AssertionError(f"Notebook execution output in source archive: {name}")
        files = {"/".join(parts) for parts in names if parts}
        for required in (
            "README.md",
            "LICENSE",
            "mf6pqc/py.typed",
            "tests/test_public_api.py",
            "examples/README.md",
            "examples/PHT3D_E01/input_data/input.pqi",
            "examples/PHT3D_E08/input_data/official_reference.npz",
            "examples/SaltLake_Brine3D/modflow_model.py",
        ):
            if required not in files:
                raise AssertionError(f"Source archive is missing {required}")
        for parts in names:
            if not parts:
                continue
            if parts[0] in {
                "docs",
                "CONTRIBUTING.md",
                "CHANGELOG.md",
                "CITATION.cff",
                "article",
                "cases",
                "references",
                "bin",
                ".release-checks",
                ".venv",
                "trash",
                ".venv-verify",
            }:
                raise AssertionError(f"Unwanted source-archive path: {parts}")
            if {"__pycache__", ".ipynb_checkpoints"}.intersection(parts) or parts[-1].endswith(
                (".pyc", ".pyo")
            ):
                raise AssertionError(f"Cache in source archive: {parts}")
            if (
                parts[0] == "examples"
                and len(parts) > 3
                and parts[2] in {"output", "simulation"}
                and parts[-1] != ".gitkeep"
            ):
                raise AssertionError(f"Generated example file in source archive: {parts}")
    print("Wheel and source archive contents passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the contents of MF6PQC release artifacts.")
    parser.add_argument("directory", type=Path)
    check_distribution(parser.parse_args().directory)
