# Releasing MF6PQC

For the Zenodo and PyPI steps, see the
[Windows PowerShell guide](publishing-first-release.zh-CN.md).

## Version and source

The version is defined in `mf6pqc/_version.py` and read statically by setuptools.
Use `1.0.0` for package and citation metadata, `v1.0.0` for the Git tag, and
MF6PQC v1.0 as the model name in the paper. Update `CITATION.cff` and
`CHANGELOG.md` with each release. Add the actual release date when publishing;
do not assign the old v0.1.0 DOI to v1.0.0.

Commit the reviewed release files and create the tag on that exact commit.
Build from a clean checkout or a reviewed source archive. Local research folders,
native binaries, notebook execution output and generated model results do not
belong in the software distribution. Preserve scientific inputs and reference
arrays. Archive paper-specific results separately when needed for reproduction.

## Checks and artifacts

```bash
python -m pip install '.[dev,examples]'
python -m unittest discover -s tests -v
python -m ruff check mf6pqc tests examples scripts
python -m ruff format --check mf6pqc tests examples scripts
python scripts/check_examples.py
python -m build
python -m twine check --strict dist/*.whl dist/*.tar.gz
python scripts/check_distribution.py dist
```

The commands above do not execute native simulations. Numerical validation is
separate; its actual scope and limitations must be reported, including the
[recorded E13 discrepancy](release-readiness.md). Install the wheel in a separate
environment and verify import from outside the source tree before uploading.

Upload the same reviewed wheel and source distribution to TestPyPI, then PyPI.
Use explicit `.whl` and `.tar.gz` filenames: source ZIP archives and checksum
files are for archiving, not for `twine upload`. Never replace an already
published version's files with different code.

## Optional GitHub publication workflow

`.github/workflows/release.yml` is manually dispatched with an existing release
tag. It checks that the tag, checked-out commit and package version agree,
executes backend-free checks, and uses PyPI Trusted Publishing. Configure the
`pypi` GitHub environment and the corresponding publisher for repository
`wangzitao21/mf6pqc`, workflow `release.yml`, environment `pypi` before using it.
Manual upload and this workflow are alternative ways to publish the same files.

Zenodo's GitHub integration is independent of PyPI publication. If enabled,
publishing a GitHub Release also triggers Zenodo archiving. Choose either that
route or a manual new-version deposit for the same software release, and confirm
that v1.0.0 belongs to the existing v0.1.0 version series.
