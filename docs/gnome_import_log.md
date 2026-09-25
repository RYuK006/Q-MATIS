> [!CAUTION]
> **SUPERSEDED** — This import was reverted due to CC BY-NC 4.0 licensing incompatible with intended commercial use. See `docs/gnome_rollback_log.md`.

# GNoME Import Log

**Date**: 2026-07-18
**Source**: `gs://gdm_materials_discovery/gnome_data/stable_materials_summary.csv`
**License**: CC BY-NC 4.0

## Import Statistics
- **Total Pulled**: 554,054 records
- **Duplicates Skipped**: 0 (The Lake was completely empty prior to this import)
- **Total Imported**: 554,054 records successfully inserted into `qmatis_lake.db`.

## Verification Spot-Check Results
- Final row count in `materials` table exactly matches the expected count: `554,054`.
- Extracted structure provenance fields natively support `source="GNoME_v1"`.
- Tested `is_rejected=False` flag correctly populated.
- Foreign Key mappings correctly generated for `decision_history`.
- Created `THIRD_PARTY_NOTICES` file for license attribution compliance.

The dataset is now safely embedded within the Q-MATIS infrastructure.
