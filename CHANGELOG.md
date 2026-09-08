# Changelog

## Unreleased — 0.2.0 development series

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

The current E13 cross-code discrepancy also occurs with the pre-refactor core.
Its acceptance thresholds remain unchanged; see the release verification record.
