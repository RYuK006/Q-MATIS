# Changelog

All notable changes to this project will be documented in this file.

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
