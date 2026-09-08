# Release procedure

首次发布请参照 [Windows PowerShell 中文操作指南](publishing-first-release.zh-CN.md)。

Packaging follows the [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
and [setuptools file-selection guidance](https://setuptools.pypa.io/en/stable/userguide/miscellaneous.html).
The version is defined once in `mf6pqc/_version.py` and read statically by setuptools.
No native backend is loaded to obtain package metadata.

1. Resolve the scientific blockers in `docs/release-readiness.md`. Record the
   quantitative acceptance results and cases intentionally not run.
2. Select a release number in `mf6pqc/_version.py`, update `CHANGELOG.md`, and
   review public compatibility and dependency constraints.
3. From a clean virtual environment, install `.[dev,examples]`, run unit/static
   checks, and the relevant short native checks. Use additional platforms before
   advertising native support on them.
4. Build with `python -m build`, then run `python -m twine check --strict dist/*`.
5. Inspect the wheel and sdist using `python scripts/check_distribution.py dist`.
   Install the wheel in a separate environment and import it from outside the
   checkout. Set `MF6PQC_USE_INSTALLED=1` for native example runs against that wheel.
6. Review the distributions and publish to TestPyPI before the public release.
   Test installation with normal runtime dependencies resolved from PyPI.
7. Publish the reviewed distributions to PyPI, create the matching version tag,
   and archive the validation evidence alongside the release.

The included `release.yml` is manually dispatched and uses PyPI Trusted
Publishing with a GitHub `pypi` environment. Configure its required reviewers and
the corresponding PyPI trusted publisher before enabling publication. No token
belongs in source control. The workflow rejects development versions and reruns
checks; it does not replace the scientific release review.

Do not upload the current development build as a stable release. This task
prepares artifacts locally; it does not upload or publish anything.
