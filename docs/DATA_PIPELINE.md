# Data Pipeline & Universal Materials Lake

Q-MATIS standardizes structural, physical, and historical data across domains into the append-only **Materials Lake**.

## Data Ingestion
1. **Public Databases**: Initial data is imported from sources like Materials Project, OQMD, and specialized datasets (e.g., SuperCon).
2. **Experimental Results**: Future phases support direct ingestion of laboratory validation data.
3. **Materials Knowledge Graph (QMKG)**: Rather than flat CSVs, all imported and generated data is structured relationally via the Materials Lake (SQLite/Parquet). Every compound receives a UUID.

## Graph Construction
Nodes are mapped using expansive, multi-domain physicochemical descriptors.
Edges encode spatial distances via a Gaussian RBF expansion.

## Append-Only Tracking
When data passes through the pipeline (e.g., Candidate Generation, Physics Filtering, GNN Inference), the exact result, uncertainty, and decision parameters are appended to the Scientific Memory Engine. No negative results are discarded; failures are scientifically critical for future model iterations.
