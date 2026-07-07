# Q-MATIS Roadmap

Q-MATIS is built for a decades-long scientific vision. Our long-term mission is to build the world's scientific memory for materials, creating permanent scientific infrastructure capable of supporting future AI systems, autonomous laboratories, and materials discovery across every scientific discipline.

The progression below outlines our path from foundational research prototype to a global, closed-loop Universal Materials Intelligence Platform.

## 1. Research Foundation (Completed)
Establish the core Deep Learning pipeline.
- Implement robust GNN Encoders (ALIGNN, CGCNN) for structural representations.
- Build Multi-Task Learning frameworks to cross-pollinate insights across tasks.
- Demonstrate predictive capacity on initial benchmark datasets.

## 2. Universal Materials Knowledge Graph (Completed)
Migrate from transient, fragmented data representations to a permanent storage model.
- Implement a rigid SQLite/Parquet hybrid ledger.
- Build the `MaterialEntity` data model to track materials with unique UUIDs.
- Enforce strict append-only constraints across all database schemas.

## 3. Physics-Constrained Discovery (Completed)
Ensure AI is bounded by physical reality.
- Implement the Candidate Generation Engine.
- Subject all generated materials to universal domain-knowledge filters (Charge Neutrality, Wyckoff Preservation, Bond-Valence Heuristics).
- Integrate Active Learning and epistemic uncertainty bounds via Deep Ensembles.

## 4. Fault-Tolerant Scientific Platform (In Progress)
Build OS-level robustness for continuous discovery on unreliable hardware.
- Implement the `ResearchExecutionEngine`.
- Enable multi-level resumability across experiments and batches.
- Solidify the Scientific Memory Engine so no compute cycle or generated data is ever lost during crashes.

## 5. High-Throughput Virtual Screening (HTVS)
Scale the Candidate Generation Engine across compute clusters.
- Distribute inference workloads horizontally.
- Orchestrate millions of daily candidate evaluations against the trained surrogate models.

## 6. Universal Property Prediction
Expand the platform beyond its initial benchmarks (e.g., superconductivity).
- Build generalized prediction heads for diverse properties: Formation Energy, Band Gap, Elastic Modulus, Thermal Conductivity, Catalytic Activity, Battery Voltage, Hydrogen Storage, etc.
- Seamlessly route the predictions directly into the Materials Lake.

## 7. Materials Foundation Model
Train a unified single-encoder representation.
- Consolidate all architectural learnings into a massive multi-modal Foundation Model.
- A single encoder must understand the physics of all crystalline matter, making it rapidly fine-tunable for any new, unforeseen property.

## 8. Retrieval-Augmented Scientific Memory
Augment generation processes with context from the global Materials Lake.
- Feed vast quantities of append-only historical decisions and lineage data back into the LLMs/GNNs.
- Enable the AI to read its own past failures to avoid repetitive exploration.

## 9. Closed-Loop Autonomous Discovery
Eliminate the human bottleneck in the hypothesis generation loop.
- The system independently explores chemical space.
- Autonomously schedules simulations, analyzes uncertainties, and refines its own models continuously.

## 10. AI-Assisted Laboratory Integration
Bridge virtual screening with physical laboratory synthesis.
- Integrate APIs with automated robotic synthesis labs.
- Close the loop entirely: from digital generation, to physical synthesis, physical measurement, and directly back into the Materials Lake.

## 11. Global Collaborative Materials Database
Open the ecosystem.
- Serve the Universal Materials Lake to the global public.
- Transform Q-MATIS into the foundational infrastructure underlying collaborative materials science research worldwide.
