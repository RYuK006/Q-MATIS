# GNoME Import Rollback Log

**Date**: 2026-07-18

## Reason for Rollback
The GNoME dataset structural and property data is licensed under the Creative Commons Non-Commercial (CC BY-NC 4.0) license. This is strictly incompatible with the intended commercial goals (for-profit new compound discovery) of this project. The data cannot be utilized at any stage of the pipeline and has been completely purged to avoid licensing contamination.

## Rollback Execution
- **`materials` Table**: Deleted 554,054 records where `source='GNoME_v1'`.
- **`decision_history` Table**: Deleted 554,054 records linked to the GNoME material entities.
- **Row Counts**: 
  - Before Rollback: 554,054 materials (554,054 tagged as GNoME)
  - After Rollback: 0 materials (0 tagged as GNoME) - Successfully restored to pre-import state.
- **File System Cleanup**:
  - Deleted `data/stable_materials_summary.csv`.
  - Deleted `THIRD_PARTY_NOTICES`.
  - Deleted all intermediate import and verification Python scripts in the `scratch` directory.
  - Annotated `docs/gnome_import_plan.md` and `docs/gnome_import_log.md` with a `SUPERSEDED` notice.

## Verification
No GNoME-derived data, cached CSVs, or structural representations remain in the Materials Lake or anywhere else in the local repository workspace.
