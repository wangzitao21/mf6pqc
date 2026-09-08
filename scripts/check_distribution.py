import argparse
import tarfile
import zipfile
from pathlib import Path


def check_distribution(directory: Path) -> None:
    wheels = list(directory.glob("*.whl"))
    sources = list(directory.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise AssertionError(
            "Use a clean dist directory containing exactly one wheel and one source archive"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        if "mf6pqc/__init__.py" not in names or "mf6pqc/py.typed" not in names:
            raise AssertionError("Missing package entry point or type marker")
        if any(not (n.startswith("mf6pqc/") or ".dist-info/" in n) for n in names):
            raise AssertionError("Wheel contains files outside the runtime package and metadata")
    with tarfile.open(sources[0]) as archive:
        names = [Path(n).parts[1:] for n in archive.getnames()]
        files = {"/".join(parts) for parts in names if parts}
        for required in (
            "README.md",
            "LICENSE",
            "mf6pqc/py.typed",
            "tests/test_public_api.py",
            "docs/release-verification.json",
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
            if "__pycache__" in parts or parts[-1].endswith((".pyc", ".pyo")):
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
