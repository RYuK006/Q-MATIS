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
[Experiment 2: Formation Energy](#experiment-2-scientific-memory-hypothesis-formation-energy) •
[Experiment 3: Phonon Frequency](#experiment-3-generalization-test-phonon-frequency) •
[What We Now Know](#what-we-now-know-across-both-experiments) •
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

Two controlled experiments have now been completed, across two independent physical properties. Both are summarized below in full, including the confounds discovered and corrected along the way.

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
- Statistical evaluation utilities (paired t-tests, confidence intervals, win/loss tallies)
- Structure deduplication cache (`experiments/shared/dedup_cache.py`), using `pymatgen.analysis.structure_matcher.StructureMatcher` to prevent redundant candidate generation

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

The purpose of this infrastructure is **reproducibility**. Whether this accumulated information improves future learning has now been tested twice — see below.

## Engineering Principles

- Append-only records whenever practical.
- Reproducible experiments, with raw per-seed results saved alongside every summary statistic.
- Versioned configurations.
- Multi-seed evaluation.
- Explicit statistical reporting, including comparisons that don't support the hypothesis being tested.
- Negative results are preserved instead of discarded.
- Claims must follow evidence rather than precede it.

---

# Experiment 2: Scientific Memory Hypothesis (Formation Energy)

## Hypothesis
Rejected materials may improve downstream prediction if the model learns **why** they were rejected.

## Design
Four training configurations were compared on formation-energy prediction (CGCNN, Matbench-derived dataset), using a preregistered protocol, identical dataset splits, 10 random seeds, paired statistical testing, and explicit negative controls:

| Run | Description |
|---|---|
| **A** | Baseline (accepted materials only) |
| **B1** | Accepted + rejected, zero-gradient control |
| **B2** | Accepted + rejected, auxiliary rejection-reason supervision |
| **C** | Accepted + rejected, scrambled rejection-reason labels |

## Result: Not Supported
Run B2 showed a higher mean than baseline, but the difference was not statistically significant (p = 0.17, CI crossing zero). More importantly, **B2 did not outperform B1** — it lost to the zero-gradient control on 7 of 10 seeds. Whatever benefit existed could not be attributed to rejection-reason supervision.

## The Follow-Up Mystery, and Its Resolution
Run B1 itself beat the original baseline (Run A) with p = 0.009 — a real effect, but an unexplained one, since B1 receives zero gradient signal from rejected structures. The leading hypothesis was that mixing rejected structures into batches increased the number of gradient steps per epoch (~164 in B1/B2/C vs. ~125 in A), and that this — not any information content — explained the gap.

**This was tested directly.** A matched-step-count control (Run A′: identical to A, batch size reduced so its step count matched B1/B2/C) was run across the same 10 seeds. Result:

- **Run A′**: 0.3525 ± 0.0285 Test MAE
- **Run B1**: 0.3537 ± 0.0194 Test MAE
- **Mean difference**: 0.0012 (95% CI: [-0.0199, 0.0224]), p = 0.898 — statistically indistinguishable.

**The batch-dynamics explanation is confirmed.** Run B1 did not benefit from any implicit regularization or "memory" of rejected structures — it simply took more gradient steps. Extending the comparison further: A′ is also statistically indistinguishable from B2 (p = 0.377) and C (p = 0.877), and B1/B2/C are mutually indistinguishable from each other (all p > 0.25). No configuration involving rejected-structure data — with or without auxiliary supervision — showed a detectable effect once baseline training was correctly step-matched.

*(Methodology notes: `drop_last=False` was confirmed identical across all DataLoaders. Loss averaging was confirmed to use only accepted-sample denominators, ruling out an implicit gradient-scaling confound. A′ showed the highest variance among the five arms (std = 0.0285), making it the noisiest-behaving arm alongside being the best-performing one.)*

---

# Experiment 3: Generalization Test (Phonon Frequency)

## Purpose
Test whether Experiment 2's null result generalizes to a different physical property, or was specific to formation energy.

## Property
Weighted last-phonon-DOS peak frequency (cm⁻¹) from the `matbench_phonons` dataset — a vibrational/lattice-dynamics quantity related to, but distinct from, thermal conductivity.

## Design Corrections Applied From the Start
This experiment incorporated every lesson from Experiment 2 before training began, rather than discovering them after the fact:

- The step-matched A′ baseline was used as the baseline from the outset (no naive unmatched baseline was run first).
- Dataset size (1,265 structures total — far smaller than Experiment 2's ~16,000) was addressed by increasing seeds to 20 and fixing the train/test split (1,012/253) once, reused identically across all seeds, so seed variation isolated model/training variance rather than compounding it with split-sampling variance.
- A convergence pilot caught an undertrained baseline at the originally-planned 10 epochs; epoch count was increased to 50 after confirming convergence against a held-out validation split.
- A test-set leakage flaw discovered during this process — checkpoints had implicitly been selected using test performance rather than a genuine holdout — was corrected here (validation-based early stopping) and retroactively flagged in `FINDINGS_v2.md` and `FINDINGS_v3.md`, since the same flaw likely affected Experiment 2's absolute (but probably not relative) MAE values.

## Result: Inconclusive / Not Supported

| Arm | Mean MAE | Std (ddof=0) |
|---|---|---|
| A′ (baseline) | 128.35 | 14.17 |
| B1 (rejected, no aux) | 125.10 | 10.05 |
| B2 (rejected + aux) | 123.61 | 11.43 |
| C (rejected + scrambled aux) | 124.51 | 13.16 |

**B2 vs A′ (primary test):** +3.69% relative improvement (clears the preregistered 3% threshold), but 95% CI crosses zero asymmetrically ([-1.60, +11.07]), p = 0.134, and the win/loss split across 20 seeds was a near coin-flip (11/9). This result is honestly labeled **INCONCLUSIVE**, not a confident null — the same rigor applied to false positives elsewhere in this project applies here to false negatives too.

**B1 / B2 / C mutual-comparison matrix:** all three pairwise comparisons among the augmented arms show ≤1% relative difference and p > 0.50. This part of the result is decisively null — the augmented arms are statistically indistinguishable from each other regardless of whether the rejection-reason label is real, scrambled, or absent entirely.

A′'s comparatively high variance (std = 14.17) appears partly driven by a handful of seeds (42, 48, 57, 59) where the smaller-batch baseline trained less stably — plausibly an artifact of reduced batch size increasing gradient noise, rather than evidence that rejected-data arms provide a real benefit.

**Overall outcome: NOT SUPPORTED**, on the compound preregistered criterion. The primary comparison is inconclusive rather than cleanly null, but the mutual-comparison matrix rules out the auxiliary rejection-reason signal specifically as the source of any effect — consistent with, though less decisive than, Experiment 2's result.

---

# What We Now Know, Across Both Experiments

Two independent, preregistered, confound-corrected experiments — on two unrelated physical properties, using two different dataset scales — have both failed to find support for the hypothesis that auxiliary rejection-reason supervision improves downstream property prediction:

- **Formation energy** (N ≈ 16,000, 10 seeds): decisively not supported. Every rejected-data configuration was statistically indistinguishable from a correctly step-matched baseline.
- **Phonon frequency** (N = 1,265, 20 seeds): not supported on the compound criterion; the primary comparison is inconclusive rather than cleanly null, but the mutual-comparison matrix rules out the auxiliary-label mechanism specifically.

This eliminates one specific, narrow formulation of the "Scientific Memory" hypothesis:

> Explicit rejection-reason labels provide useful auxiliary supervision for downstream property prediction.

It does **not** eliminate the broader question of whether rejected structures can contribute to learning through some other mechanism (self-supervised pretraining, contrastive learning, retrieval-based approaches). Those remain open, untested hypotheses — see below.

---

# Research Philosophy

The primary objective of Q-MATIS is not to defend architectural ideas. It is to test them.

Whenever possible, new ideas are evaluated using:

- preregistered hypotheses
- controlled baselines
- negative controls
- multiple random seeds
- statistical significance testing
- public reporting of positive and negative results

If an experiment contradicts an earlier design assumption, the assumption changes. The codebase exists to support that process. Infrastructure is not built to justify a predetermined conclusion — it is built to make conclusions testable. A negative result that eliminates an incorrect hypothesis is treated as a successful outcome, not a setback to be minimized in the writeup.

Two confounds were discovered mid-experiment during this process (a batch-dynamics artifact in Experiment 2, a test-set leakage flaw discovered while designing Experiment 3) and corrected transparently, with prior findings retroactively caveated rather than silently left as-is.

---

# Open Research Questions

The following remain genuinely untested. Question 5 from earlier iterations of this document (why did the zero-gradient control beat baseline?) has been resolved above and removed from this list.

## Question 1
**Can rejected structures improve learned representations without explicit rejection labels?**
Possible approaches: contrastive learning, masked graph modeling, self-supervised pretraining. This is the most promising remaining direction given both experiments above ruled out the *label-based* mechanism specifically, without testing label-free mechanisms.

## Question 2
**Does complete experiment provenance improve retrieval or downstream reasoning?**
Possible evaluation: retrieval benchmarks, experiment recommendation, reproducibility analysis.

## Question 3
**Which information is actually worth preserving?**
Future experiments will evaluate individual components (optimizer history, model checkpoints, failed candidates, DFT outputs, uncertainty estimates, experiment lineage) rather than assuming all metadata is equally valuable.

## Question 4
**Can one encoder support multiple materials-property prediction tasks without degrading performance?**
Evaluation will begin using established public benchmarks before expanding toward broader multi-task settings.

---

# Long-Term Vision

The long-term direction of Q-MATIS is intentionally separated from its verified capabilities.

If the underlying hypotheses continue to survive experimental testing over the coming years, the project may gradually evolve toward:

- a reusable materials representation model
- unified storage of computational and experimental materials data
- reusable datasets built from accumulated experimental history
- AI-assisted materials discovery workflows

These are **research ambitions**, not current capabilities. Their feasibility depends entirely on future experimental evidence — including the open questions above, all of which remain untested as of this writing.

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
