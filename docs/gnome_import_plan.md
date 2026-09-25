> [!CAUTION]
> **SUPERSEDED** — This import was reverted due to CC BY-NC 4.0 licensing incompatible with intended commercial use. See `docs/gnome_rollback_log.md`.

# GNoME Stable-Materials Import Plan

## 1. Exact Source URL
The GNoME dataset is officially hosted by Google DeepMind on Google Cloud Storage at `gs://gdm_materials_discovery/`. 
The primary summary file (containing formulas, energies, and metadata) is located at `https://storage.googleapis.com/gdm_materials_discovery/stable_materials_summary.csv`. The actual CIF structure files are packed in `.zip` archives in the same bucket. We will download the summary CSV and a subset/batch of structure ZIPs.

## 2. License and Attribution
WARNING: The prompt assumed the license is **Apache 2.0**, but my research confirms the GNoME dataset is released under the **CC BY-NC 4.0** (Creative Commons Non-Commercial) license. The DeepMind *code* might be Apache 2.0, but the *data* is CC BY-NC. 
We will satisfy the attribution requirement by creating a `THIRD_PARTY_NOTICES` file in the repository root explicitly stating the CC BY-NC 4.0 license, linking to the DeepMind repository, and explicitly declaring that this data is for non-commercial research use only.

## 3. Field Mapping
Based on the known GNoME dataset format, we will map it as follows:
- **GNoME Data** -> **`MaterialEntity`**:
  - Structure (CIF) -> `structure_json` (Parsed via `pymatgen` and converted to dict)
  - Formula -> `formula` and `reduced_formula`
  - `source` -> `"GNoME_v1"`
  - `is_rejected` -> `False`
  - Formation Energy / Stability metrics -> `metadata` dict or `properties` dict.
- **`DecisionRecord`**:
  - `action="Imported"`
  - `reason="GNoME stable materials public release"`
  - `responsible_module="GNoME_Batch_Importer"`

## 4. Provenance Tagging
As verified in Chunk 0, the `MaterialEntity` schema natively supports a `source` string field. All imported GNoME structures will have `source="GNoME_v1"`.

## 5. Deduplication Approach
We will write a lightweight standalone Python script for this import. Before inserting a structure, we will query `qmatis_lake.db` for materials with the identical `reduced_formula`. If any exist, we will load their `structure_json` into `pymatgen.core.Structure` and use `pymatgen.analysis.structure_matcher.StructureMatcher` to test equivalence. If a match is found, the GNoME candidate is skipped as a duplicate.
