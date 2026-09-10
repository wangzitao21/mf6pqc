# Changelog

## 1.0.0

- Provide SNIA, SIA and Strang coupling with optional porosity, hydraulic
  conductivity, diffusion and density feedback, and an opt-in thermal pathway.

- Require modflowapi 1.0.1 or a compatible 1.x release.
- Standard Python logging with opt-in progress messages for library users.
- Single-source versioning, explicit package contents, build checks and CI.
- Consistent example layout, portable paths, isolated native verification,
  import-safe drivers and clean plotting notebooks.
- Strict integer, schedule, mask, initial-condition and tolerance validation.
- Native PhreeqcRM status failures become actionable Python exceptions;
  failures finalize resources and closed instances cannot be reused.
- Correct geometric TDIS expansion for multipliers close to one; reject
  unsupported time units and ATS before native initialization.
- Honor configured chemistry print masks and initial user density.
- Read calculated density independently of selected output, preserving all columns.
- Validate serialized metadata and physical fields before writing; write the
  manifest last and include provenance and explicit time-axis semantics.
- Retain historical class, constructor, coupling-method and updater imports.

The recorded E13 pH/Ca comparison exceeds its acceptance thresholds; see
[verification history](docs/release-readiness.md). Version 1.0.0 packaging
preparation does not constitute a new numerical validation of the examples.

## 0.1.0 — 2026-03-01

- Initial archived release on [Zenodo](https://doi.org/10.5281/zenodo.18822578).
