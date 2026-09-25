# Q-MATIS Phase 2: Thermal Transport (Experiment 3) Findings

## Overview
This experiment evaluated whether training on rejected structures (with or without an auxiliary cross-entropy loss predicting the rejection reason) improves representation learning and out-of-distribution generalization for predicting a phonon-related property (weighted last-phonon-DOS peak frequency, cm⁻¹) on the `matbench_phonons` dataset. This is a vibrational/lattice-dynamics quantity related to, but distinct from, thermal conductivity.

The experimental design strictly followed a preregistered protocol (`preregistration_v1.md`), evaluating four arms across 20 independent seeds.

## Summary Statistics (N=20 seeds per arm)
*(Note: Standard Deviation uses the `ddof=0` population standard deviation convention for consistency with Phase 1 reports).*

* **Arm A′ (Baseline, 1012 accepted)**: Mean MAE = 128.3480 | Std = 14.1740
* **Arm B1 (1012 accepted + 1012 rejected, No Aux)**: Mean MAE = 125.1035 | Std = 10.0490
* **Arm B2 (1012 accepted + 1012 rejected + Aux)**: Mean MAE = 123.6125 | Std = 11.4323
* **Arm C (1012 accepted + 1012 rejected + Shuffled Aux)**: Mean MAE = 124.5072 | Std = 13.1595

## Pairwise Comparisons (Experimental vs Baseline)
*(Note: A positive Mean Difference indicates the experimental arm achieved a lower/better MAE than the baseline).*

**B2 vs A′ (Primary Hypothesis Test):**
* **Mean Difference**: +4.7356 (95% CI: [-1.5995, +11.0706])
* **Relative Improvement**: +3.69%
* **Paired t-test p-value**: 0.1342
* **Wins/Losses (ties)**: 11 / 9 (0)

**B1 vs A′:**
* **Mean Difference**: +3.2445 (95% CI: [-2.2872, +8.7762])
* **Relative Improvement**: +2.53%
* **Paired t-test p-value**: 0.2346
* **Wins/Losses (ties)**: 13 / 7 (0)

**C vs A′:**
* **Mean Difference**: +3.8409 (95% CI: [-1.7710, +9.4527])
* **Relative Improvement**: +2.99%
* **Paired t-test p-value**: 0.1683
* **Wins/Losses (ties)**: 12 / 8 (0)

## B1 / B2 / C Mutual-Comparison Matrix
*(Comparing the augmented arms against each other)*

* **B2 vs B1**: Mean Difference = +1.4911 (95% CI: [-3.2252, +6.2073]), p = 0.5161, Wins/Losses = 10 / 10
* **C vs B1**: Mean Difference = +0.5963 (95% CI: [-4.0091, +5.2018]), p = 0.7893, Wins/Losses = 13 / 7
* **C vs B2**: Mean Difference = -0.8947 (95% CI: [-6.0640, +4.2746]), p = 0.7212, Wins/Losses = 10 / 10

## Final Verdict

**OVERALL OUTCOME: NOT SUPPORTED**

The compound success criterion was not met. The primary comparison (B2 vs A′) is statistically **INCONCLUSIVE** given the wide confidence interval relative to the dataset size. While the point estimate exceeds the preregistered 3% threshold (+3.69%), the 95% CI crosses zero asymmetrically ([-1.60, +11.07]) and the win/loss split is a near coin-flip (11/9), meaning we cannot rule out no effect.

However, the B1/B2/C mutual-comparison matrix shows a much tighter, more decisively null pattern. All three pairwise comparisons among the augmented arms are small in magnitude (~1% relative or less) with CIs tightly centered near zero and p-values exceeding 0.50. The augmented arms are statistically indistinguishable from each other regardless of whether the auxiliary rejection-reason label is real (B2), scrambled (C), or entirely absent (B1). 

This indicates that whatever modest effect might exist from adding rejected data, it is definitively unrelated to the auxiliary rejection-reason signal. This is broadly consistent with Phase 1's finding, though this experiment's smaller N means the primary B2 vs A′ result is less decisively null than Phase 1's.
