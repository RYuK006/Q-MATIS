# Preregistration v1: Experiment 3 - Thermal Transport (Phonons)

This experiment tests whether the null finding from Phase 1 (formation energy) generalizes to a different physical property, thermal transport (phonon frequency peak).

## Hypothesis
**Hypothesis under test:** Does preserving rejected/unstable candidate structures — with or without auxiliary rejection-reason supervision — provide a measurable benefit to a thermal-transport property predictor, beyond what a correctly step-matched baseline achieves?

Given Phase 1's result, where rejected data showed zero detectable benefit for formation energy under any tested formulation, this experiment explicitly tests whether that null result generalizes to a different property. We are not re-litigating formation energy; we are determining if the failure of the "Scientific Memory" hypothesis holds across diverse learning tasks.

## Property & Architecture
- **Property:** Thermal transport, specifically the weighted phonon frequency peak (cm^-1) from the `matbench_phonons` dataset (Matbench v0.1).
- **Architecture:** Continuous Filter Convolutional Neural Network (CGCNN), reused unchanged from Phase 1.

## Dataset & Deviation from Phase 1
- **Candidate pool for rejected structures:** In a deviation from the original Phase 1 design, rejected candidates are seeded by directly perturbing structures from the `matbench_phonons` training split (rather than an external GNoME pool). This ensures the parent materials share the exact same chemical domain as the target task.
- **Ratio & Variance Sampling:** We maintain a strict **1:1 ratio** of accepted to rejected structures (1,012 of each per seed). This higher concentration of rejected data is a deliberate choice to compensate for the small total dataset size ($N=1,265$) compared to Phase 1's ~3:1 ratio. Additionally, because rejected structures are generated dynamically, the exact perturbed geometries vary per seed. This is a deliberate design choice: varying the "noisy" rejected candidates across seeds prevents the model from overfitting to a single random draw of perturbations, providing a more robust estimate of their effect across the 20 seeds.
- **Dataset Size & Split:** The `matbench_phonons` dataset contains only 1,265 rows. We use a standard 80/20 train/test split, resulting in exactly **1,012 training structures** and **253 test structures**. The split is strictly randomized and deterministically seeded (saved to `split_indices.json`).
- **Internal Validation Split:** To monitor convergence without touching the pristine test set, the 1,012 training structures are internally split 80/20 (809 train, 203 validation) using the same deterministic seeding logic. This validation split is fixed and reused identically across all arms and seeds. The 253-point test set is reserved purely for the final arm-vs-arm comparison.
- **Isolating Variance:** To isolate model initialization/shuffling variance from split-sampling variance on this smaller dataset:
  - **Seeds:** Increased to 20 seeds (e.g., 42–61).
  - **Data Split:** The 1,012/253 train/test split will be generated *once* deterministically and reused identically across all 20 seeds. The seed will control only model initialization and data loader shuffling.
  - **Expectations:** The small test set ($N=253$) means confidence intervals for MAE will naturally be wider. This makes the statistical bar for a "supported" result appropriately higher, not lower.

## 4-Arm Experimental Design (Step-Matched from the Start)
- **Epochs and Step Count:** The training runs for exactly **50 epochs** across all arms. A 10-epoch pilot run confirmed the loss was still decreasing steeply due to the small number of steps per epoch on this smaller dataset. 50 epochs at 32 steps/epoch yields exactly 1,600 total gradient updates, mathematically matching the gradient update count of Phase 1 to ensure a fair comparison of final converged performance.
- **Learning Rate:** The learning rate is strictly fixed at `1e-3` (Adam optimizer) for all arms and is **not** scaled by batch size, ensuring no hidden hyperparameter confounds between arms.
- **Run A′ (baseline):** Accepted-only. The batch size will be pre-calculated and matched to the exact gradient step count that the mixed-batch arms will use, preventing the confound discovered in Phase 1.
- **Run B1 (zero-gradient control):** Accepted + rejected, no auxiliary loss.
- **Run B2 (auxiliary supervision):** Accepted + rejected, auxiliary rejection-reason CrossEntropy loss ($\lambda=0.5$).
- **Run C (scrambled control):** Accepted + rejected, rejection-reason labels shuffled.

## Leakage Guard
Rejected/perturbed candidates are strictly generated *only* from structures assigned to the training split. No candidate will be seeded from the test split. This will be programmatically asserted during generation.

## Success Criterion
The hypothesis is considered **SUPPORTED** only if the compound criterion is met:
1. **B2 beats A′** by $\ge 3\%$ relative MAE improvement. Statistical significance is established via the paired-difference 95% CI (computed from the per-seed matched differences) excluding zero, consistent with the method used in Phase 1's analysis scripts. This is the sole significance criterion; individual arms' raw CIs overlapping or not is not itself a criterion.
2. **B2 outperforms B1**, establishing that the gain isn't explained by data volume/step count alone.
3. **B2 outperforms C**, establishing that the gain isn't generic multi-task regularization on arbitrary labels.

If B2 is statistically indistinguishable from A′, B1, or C, the null hypothesis is **ACCEPTED**, reinforcing Phase 1's conclusion.
If the 95% CI is wide enough that statistical significance cannot be established either way (i.e. the confidence interval crosses the baseline but the point estimate is large, or vice versa), the result is **INCONCLUSIVE**, not supported.
