# Changelog

All notable changes to this project will be documented in this file.

## [v1.1.0-alpha] - 2026-07-07
### Strategic Pivot
- **Universal Materials Intelligence Platform**: Q-MATIS vision expanded from a superconductor predictor to a universal AI operating system for materials discovery, establishing a permanent Scientific Memory Engine.
### Added
- **Materials Lake**: Hybrid SQLite/Parquet append-only ledger tracking all materials, predictions, and physical properties.
- **Scientific Memory**: Complete provenance tracking across generated candidates, failed structures, and physics filter logic.
- **Fault-Tolerant Research Engine**: OS-level multi-stage resumability for running extensive HPC pipelines without losing progress.
- **Physics-Aware Discovery Engine**: Strict domain constraints (charge neutrality, Wyckoff preservation) applied prior to ML inference.

## [v1.0.0-alpha] - 2026-07-04
### Added
- **Repository Architecture**: Complete restructuring of Q-MATIS into a research-grade open-source platform.
- **Data Pipeline**: `DataOrchestrator` fusing NIMS SuperCon properties with Materials Project 3D structures via SQLite caching.
- **Graph Engineering**: PyTorch Geometric integration encoding 92-dimensional atomic descriptors and RBF edge representations.
- **Models**: `CGCNN` baseline and `ALIGNN` native PyG edge-gated convolutions.
- **Training**: Pretraining, Transfer Learning, and Multi-Task Learning pipelines for $T_c$ and Formation Energy.
- **Deep Ensembles**: Epistemic uncertainty calibration providing PICP and MPIW metrics.
- **Active Learning**: Candidate substitution engine with Upper Confidence Bound (UCB) ranking.
- **Analysis**: Complete benchmarking suite including UMAP/t-SNE latent embeddings and calibration plotting.
- **Documentation**: Subsystem technical specs, API docs, and GitHub templates.
