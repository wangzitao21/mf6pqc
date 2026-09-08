# Contributing

Install `python -m pip install -e ".[dev,examples]"` in an isolated environment.
Run unittest, Ruff lint/format checks, and `python scripts/check_examples.py`
before submitting changes. Use an appropriate short native case when changing
numerical behavior; do not launch expensive benchmarks automatically.

Keep the runtime independent of individual case names, repository paths and
plotting dependencies. Add a regression test for a demonstrated numerical or
lifecycle failure. State units, array ordering, timestep semantics and state
ownership whenever an interface changes. Preserve reference inputs and acceptance
thresholds during refactoring; a changed benchmark definition requires evidence
and explicit documentation.

Examples follow the shared layout described in `examples/README.md`. Put model
construction and scientific constants in `modflow_model.py`, orchestration in
`run.py`, and figures in `plot.ipynb` or `plot.py`. Importing a script must not
run a simulation or read generated outputs. Clear notebook outputs before commit.
Generated output and workspaces are never committed.

A pull request should describe the problem, resulting behavior, relevant tests,
quantitative before/after comparisons, and any unresolved scientific limitation.
Do not present unrun cases as newly validated. For research use, report both solver
convergence and comparison against independent evidence.
