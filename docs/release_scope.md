# Release Scope

This release keeps raw inputs, final generated/healed geometry,
tetrahedral meshes, mesh validation, and compact audit reports.
Intermediate prompt traces, large execution JSON logs, and duplicate
visualization formats are excluded unless they are the only compact
evidence for a result.

The acceptance language is intentionally conservative:

- A full repaired-body tetrahedral mesh is required before a VPMR or
  Thingi10K case is called a full-body mesh success.
- Componentwise, isolated, split, or matching-only mesh artifacts are
  diagnostic evidence and are not labeled as global success.
- Quality caveats are kept in the per-dataset manifests.

Build script: `scripts/build_dataset_release.py`
