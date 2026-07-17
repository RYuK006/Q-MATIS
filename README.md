# Q-MATIS: Quantum Materials Intelligence System

<div align="center">
  <img src="assets/logo.png" alt="Q-MATIS Logo" width="80%">
</div>

<br/>

<div align="center">
  <strong>An experimental platform for testing ideas in AI-driven materials discovery.</strong>
</div>

<br/>

<div align="center">

[What Q-MATIS Is Today](#what-q-matis-is-today) •
[Built & Verified](#built--verified) •
[Experiment 2](#experiment-2-scientific-memory-hypothesis) •
[Research Philosophy](#research-philosophy) •
[Open Research Questions](#open-research-questions) •
[Long-Term Vision](#long-term-vision) •
[Installation](#installation)

</div>

---

# What Q-MATIS Is Today

Q-MATIS is an **experimental research platform** for AI-assisted materials science.

The project focuses on two complementary goals:

1. Building reproducible infrastructure for graph neural network experiments on crystalline materials.
2. Testing falsifiable hypotheses about whether preserving additional scientific context (negative results, decision history, provenance, intermediate reasoning) improves future machine learning models.

The emphasis is deliberately on **experimentation before architecture**. Every major design decision is expected to survive controlled experiments before it becomes part of the platform.

Q-MATIS is **not** a universal materials foundation model, an AI operating system, or a proven scientific memory system today. Those remain long-term research directions rather than established capabilities.

---

# Built & Verified

The following components exist in the repository today.

## Machine Learning Infrastructure

- CGCNN training pipeline
- PyTorch Geometric dataset construction
- GPU training support
- Experiment configuration management
- Checkpoint serialization
- Multi-seed experiment execution
- Statistical evaluation utilities

---

## Scientific Data Infrastructure

Q-MATIS currently includes an append-only data model capable of recording:

- materials
- prediction history
- experiment metadata
- model versions
- configuration snapshots
- checkpoints
- decision history
- provenance
- lineage
- environment metadata
- failure logs

The purpose of this infrastructure is **reproducibility**.

Whether this accumulated information improves future learning remains an open research question.

---

## Engineering Principles

Current development follows several engineering constraints:

- Append-only records whenever practical.
- Reproducible experiments.
- Versioned configurations.
- Multi-seed evaluation.
- Explicit statistical reporting.
- Negative results are preserved instead of discarded.
- Claims must follow evidence rather than precede it.

---

# Experiment 2: Scientific Memory Hypothesis

The first major scientific question investigated by Q-MATIS was:

> **Does recording why a material was rejected provide useful learning signal beyond simply having additional training data?**

---

## Hypothesis

Rejected materials may improve downstream prediction if the model learns **why** they were rejected.

This was evaluated using formation-energy prediction.

---

## Experimental Design

Four training configurations were compared.

| Run | Description |
|------|-------------|
| **Run A** | Baseline (accepted materials only) |
| **Run B1** | Accepted + rejected materials with zero-gradient control |
| **Run B2** | Accepted + rejected materials with auxiliary rejection-reason supervision |
| **Run C** | Accepted + rejected materials with randomly scrambled rejection labels |

Key design choices:

- preregistered protocol
- identical dataset splits across all four arms, per seed
- paired evaluation
- 10 random seeds
- paired statistical testing (mean difference, 95% CI, p-value)
- explicit negative controls

The inclusion of **Run B1** was particularly important because it isolated the effect of adding rejected samples from the effect of actually learning their rejection reasons. Run B1 receives zero gradient signal from rejected structures, so any change relative to Run A in this arm cannot come from information content — only from batch composition.

---

## Preregistered Success Criterion

The primary hypothesis would only be considered supported if:

1. Auxiliary supervision (Run B2) improved over the baseline (Run A) by at least the preregistered threshold.
2. Run B2 also meaningfully outperformed Run B1, demonstrating that the gain came from the rejection-reason information rather than other training effects (e.g. batch composition, effective step count).

Both conditions were required. Meeting only one does not count as support for the hypothesis.

---

## Result

**The preregistered hypothesis was not supported.**

Across ten random seeds:

| Comparison | Mean MAE Difference | 95% CI | p-value | Win/Loss (seeds) |
|---|---|---|---|---|
| Run A vs Run B2 | 0.0145 | [-0.0076, 0.0366] | 0.17 | B2 beat A: 8/10 |
| Run A vs Run B1 | 0.0245 | [0.0079, 0.0412] | 0.009 | B1 beat A: 9/10 |
| Run B1 vs Run B2 | -0.0100 | [-0.0290, 0.0089] | 0.26 | B2 beat B1: 3/10 |
| Run C vs Run B2 | — | — | 0.37 | B2 beat C: 4/10 |

Run B2's mean MAE was lower than Run A's, but this difference was **not statistically significant** (p = 0.17, confidence interval crosses zero). Taken alone, this result would already be too weak to call a positive finding.

More importantly, Run B2 **did not outperform Run B1** — in fact it lost to the zero-gradient control on 7 of 10 seeds, and lost to the scrambled-label control on 6 of 10 seeds. Run B1, which receives no information at all from rejected structures, improved over baseline with a result *more* statistically significant (p = 0.009) than Run B2's.

This means the observed A→B2 improvement cannot be attributed to rejection-reason supervision specifically. Whatever benefit exists appears to come from something present in Run B1 as well — i.e., something unrelated to the content of the rejection labels.

### Unresolved: Why Did Run B1 Also Beat Baseline?

**This is currently the most important open question raised by Experiment 2, and it has not been answered.**

A leading candidate explanation is that mixing rejected structures into training batches changes optimization dynamics — for example, more accepted-only gradient steps per epoch when batches are diluted with rejected samples. This is a plausible hypothesis, not a confirmed one. **A matched-step-count control (rerunning Run A with batch size or step count matched to the B1/B2/C configuration) has not yet been performed.** Until that control is run, "batch dynamics" should be treated as the leading hypothesis for Run B1's result, not as an established explanation.

---

## What We Actually Learned

Experiment 2 did **not** prove that Scientific Memory improves downstream models.

It also did **not** prove that Scientific Memory is useless.

The experiment falsified one specific, narrow formulation:

> Explicit rejection-reason labels provide useful auxiliary supervision for formation-energy prediction.

It eliminated that specific mechanism. It did not eliminate the broader question of whether rejected structures or decision histories can contribute to future learning through some other mechanism. Other formulations remain open, including:

- self-supervised learning from rejected structures
- contrastive representation learning
- curriculum learning
- uncertainty-aware learning
- retrieval-based approaches
- representation learning using complete experiment history

These remain hypotheses requiring independent evaluation.

---

# Research Philosophy

The primary objective of Q-MATIS is not to defend architectural ideas.

It is to test them.

Whenever possible, new ideas are evaluated using:

- preregistered hypotheses
- controlled baselines
- negative controls
- multiple random seeds
- statistical significance testing
- public reporting of positive and negative results

If an experiment contradicts an earlier design assumption, the assumption changes.

The codebase exists to support that process. Infrastructure is not built to justify a predetermined conclusion — it is built to make conclusions testable. A negative result that eliminates an incorrect hypothesis is treated as a successful outcome, not a setback to be minimized in the writeup.

---

# Open Research Questions

Current development is organized around research questions rather than feature milestones.

## Question 1

**Can rejected structures improve learned representations without explicit rejection labels?**

Possible approaches:

- contrastive learning
- masked graph modeling
- self-supervised pretraining

## Question 2

**Does complete experiment provenance improve retrieval or downstream reasoning?**

Possible evaluation:

- retrieval benchmarks
- experiment recommendation
- reproducibility analysis

## Question 3

**Which information is actually worth preserving?**

Rather than assuming every piece of metadata is valuable, future experiments will evaluate individual components such as:

- optimizer history
- model checkpoints
- failed candidates
- DFT outputs
- uncertainty estimates
- experiment lineage

## Question 4

**Can one encoder support multiple materials-property prediction tasks without degrading performance?**

Evaluation will begin using established public benchmarks before expanding toward broader multi-task settings.

## Question 5 (raised directly by Experiment 2)

**Is the Run A vs Run B1 improvement a real optimization effect, or a batch-composition artifact?**

Requires a matched-step-count control run before any of Questions 1–4 can be interpreted cleanly, since any future experiment that mixes rejected structures into training batches will inherit this same unresolved confound.

---

# Long-Term Vision

The long-term direction of Q-MATIS is intentionally separated from its verified capabilities.

If the underlying hypotheses continue to survive experimental testing over the coming years, the project may gradually evolve toward:

- a reusable materials representation model
- unified storage of computational and experimental materials data
- reusable datasets built from accumulated experimental history
- AI-assisted materials discovery workflows

These are **research ambitions**, not current capabilities. Their feasibility depends entirely on future experimental evidence, including the still-unresolved questions raised above.

---

# Repository Structure

```
Q-MATIS/
├── datasets/
├── experiments/
├── models/
├── superconductor/
├── tests/
├── results/
├── docs/
└── README.md
```

---

# Installation

```bash
git clone https://github.com/RYuK006/Q-MATIS.git

cd Q-MATIS

python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

# Citation

If you use Q-MATIS in your research, please cite:

```bibtex
@software{q_matis_2026,
  author    = {Q-MATIS Contributors},
  title     = {Q-MATIS: An Experimental Platform for AI-Driven Materials Discovery},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/RYuK006/Q-MATIS}
}
```

---

# License

This project is released under the repository's LICENSE file.
