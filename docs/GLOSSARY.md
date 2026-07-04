# Q-MATIS Master Glossary

## Q-MATIS Development Milestones

### Milestone A1
**Project Initialization & Architecture Refactoring**
The foundational restructuring of the original prototype into a robust, object-oriented pipeline. This phase established the `DiscoveryPipeline` class, enabling reproducibility through centralized YAML configurations, deterministic random seed seeding, and proper decoupled class abstractions.

### Milestone A2
**Modular Data Layer & Caching**
Implementation of the `DataOrchestrator` to standardise inputs from disparate databases (Materials Project, SuperCon). Crucially, this milestone introduced the SQLite caching mechanism to rapidly serialize and deserialize 3D structures and prevent expensive redundant API calls during pipeline execution.

### Milestone A3
**Transfer Learning & Pretraining**
The separation of the neural network into an independent `BaseCrystalEncoder` backbone and multi-task prediction heads. This allowed Q-MATIS to pretrain representations on massive structural datasets (like the Materials Project) and transfer those topological weights to fine-tune on scarce target labels (like $T_c$).

### Milestone A4
**Multi-Task Learning**
Integration of joint-optimization paradigms, enabling the network to predict multiple properties simultaneously (e.g., Critical Temperature and Formation Energy). This milestone mathematically proved that auxiliary property learning acts as a strong regularizer, yielding superior metrics compared to single-task regressions.

### Milestone A5
**Advanced Architectural Benchmarks (CGCNN vs ALIGNN)**
The introduction of the `EncoderRegistry` and the rigorous architectural comparison between standard node-based message passing (CGCNN) and edge-gated line graph convolutions (ALIGNN), establishing ALIGNN as the superior geometric encoder for the pipeline.

---

## Core Concepts & Architectures

### CGCNN (Crystal Graph Convolutional Neural Networks)
A pioneering graph neural network architecture representing atoms as nodes and bonds as edges. It performs standard message passing, where node features are iteratively updated by aggregating the latent features of their immediate spatial neighbors, making it highly effective for bulk crystal property prediction.

### ALIGNN (Atomistic Line Graph Neural Network)
An advanced state-of-the-art architecture that constructs a "line graph" (where bonds become nodes and bond-angles become edges). By performing edge-gated message passing simultaneously on the original atomic graph and the line graph, ALIGNN elegantly captures complex multi-body interactions and bond-angle geometries.

### Transfer Learning
A machine learning technique where a model developed for a data-rich task (e.g., predicting formation energy across 150,000 MP materials) is reused as the starting point for a data-scarce task (e.g., predicting $T_c$ for 2,000 superconductors). It leverages pre-learned representations of atomic physics to improve generalization.

### Deep Ensembles
A technique to quantify epistemic uncertainty by training multiple identical neural networks with different random initializations (and optionally varying data splits). The variance between their predictions is used as a confidence bound, allowing the system to know *when it doesn't know*.

### Multi-Task Learning
A paradigm where a single neural network is trained to minimize a combined loss function across multiple target properties simultaneously. Sharing latent representations across tasks forces the model to learn fundamental physical invariants rather than overfitting to a single label.

### Active Learning
A sequential discovery loop where the model uses its own uncertainty metrics to select the most informative new candidates to evaluate. By focusing computational resources (like DFT) on high-uncertainty or high-reward regions of chemical space, it maximizes discovery throughput.

### Materials Project (MP)
A massive open-access database providing computed properties (via Density Functional Theory) for hundreds of thousands of materials. In Q-MATIS, the MP serves as the primary ground-truth source for stable 3D crystal structures and formation energies.

### SuperCon (NIMS)
The National Institute for Materials Science (NIMS) Superconducting Materials Database. It is the world's most comprehensive catalog of experimentally measured critical temperatures ($T_c$). While it provides formula and $T_c$ pairs, it often lacks complete 3D structural data, making the `DataOrchestrator` fusion with the Materials Project essential.
