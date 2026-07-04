# Q-MATIS Development Roadmap

The ultimate objective of Q-MATIS (Quantum Materials Intelligence System) is to become a fully autonomous AI-driven materials discovery platform capable of generating, evaluating, ranking, validating, and continuously improving its search for experimentally realizable high-temperature superconductors.

## Completed Milestones (Phase 0 Foundation)
- **Data Architecture**: SQLite determinism and modular MP/SuperCon orchestrators.
- **Graph Engineering**: PyTorch Geometric representations with atomic feature injection.
- **Architectures**: CGCNN baseline and native PyG ALIGNN Edge-Gated implementation.
- **Training Systems**: Multi-task learning, Transfer learning, and Deep Ensemble uncertainty calibration.
- **Platform Integrity**: Open-source GitHub publication, CI/CD, and technical documentation.

## Phase 1: Physics-Aware Candidate Generation
*Goal: Reject chemically impossible candidates before GNN inference.*
- [ ] Oxidation-state validation
- [ ] Charge neutrality constraints
- [ ] Ionic radius compatibility & electronegativity checks
- [ ] Coordination environment symmetry preservation
- [ ] Advanced alloy, defect, and vacancy generation

## Phase 2: High-Throughput Virtual Screening (HTVS)
*Goal: Scale inference to millions of candidate materials.*
- [ ] Batched inference and multi-GPU parallelism
- [ ] Asynchronous processing and checkpoint recovery
- [ ] Large-scale Parquet/SQLite storage of predictions ($T_c$, FE, novelty, uncertainty)

## Phase 3: Discovery Ranking Engine
*Goal: Multi-objective candidate scoring.*
- [ ] Combine $T_c$, uncertainty, stability (FE), and synthesis feasibility into a unified utility function.

## Phase 4: Novelty Detection
*Goal: Prioritize materials unlike anything currently known.*
- [ ] Latent-space novelty estimation (Nearest-neighbor search, clustering).
- [ ] Latent space visualization dashboards.

## Phase 5: Automated DFT Validation
*Goal: Close the physical simulation loop.*
- [ ] VASP and Quantum ESPRESSO wrappers.
- [ ] Automated generation of simulation directories.
- [ ] Parsers to ingest DFT results back into the training corpus.

## Phase 6: Autonomous Discovery Loop
*Goal: Full autonomy.*
- [ ] Continuous orchestration loop: Generate -> Filter -> Predict -> Rank -> DFT -> Retrain -> Repeat.

## Phase 7: Generative Crystal Design
*Goal: Create novel crystal topologies beyond simple substitutions.*
- [ ] Integration of Graph Variational Autoencoders (Graph VAEs).
- [ ] Integration of 3D Crystal Diffusion Models.

## Phase 8: Universal Foundation Models
*Goal: Leverage massive self-supervised physics representations.*
- [ ] M3GNet and CHGNet integration as feature extractors and fine-tuning backbones.
