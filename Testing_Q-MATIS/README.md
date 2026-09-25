# Q-MATIS H1 001

## Question 5: Why Did Run B1 Also Beat Baseline?

**Resolved Outcome:**
In Experiment 2 (v2), the zero-gradient sanity control (Run B1) outperformed the baseline (Run A) with statistical significance ($p=0.009$). The initial hypothesis was that the auxiliary task (predicting rejection reasons) provided a representation learning benefit (Run B2), but the B1 result suggested an alternative mechanism was at play.

To test this, we conducted a matched-step-count control experiment (Run A'). Run A' was identical to Run A (accepted-only, zero auxiliary loss), but its batch size was reduced from 128 to 98 to match the ~164 gradient steps per epoch taken by the mixed batches in Runs B1, B2, and C.

**Results of the Matched-Step-Count Control:**
- **Run A' (Matched Baseline, ~164 steps)**: 0.3525 ± 0.0285 Test MAE
- **Run B1 (Zero-grad Control, ~164 steps)**: 0.3537 ± 0.0194 Test MAE
- **Run A (Original Baseline, ~125 steps)**: 0.3782 ± 0.0216 Test MAE

Comparing Run A' to Run B1 yields a mean difference of 0.0012 (95% CI: [-0.0199, 0.0224]) with $p=0.8976$, making them statistically indistinguishable. The entire gap between the original baseline and Run B1 is fully explained by the increased number of gradient updates per epoch on the accepted material data. 

**Conclusion:**
The batch-dynamics/step-count explanation is **CONFIRMED**. Run B1 did not benefit from any implicit regularization or "memory" of the rejected structures; it simply took more gradient steps. As a result, the auxiliary task (Run B2) provides no significant benefit over a properly scaled baseline training regime. 

Additionally, A′ is statistically indistinguishable from B2 (p=0.377) and C (p=0.877), and B1/B2/C are mutually indistinguishable from each other (all p>0.25). This means no configuration involving rejected-structure data — with or without auxiliary supervision — shows a detectable effect once baseline training is correctly step-matched.

*(Methodology Note: `drop_last` was verified to be identical (defaulting to False) across all DataLoaders, meaning partial batches were processed identically across A, A′, and the mixed arms. It is also worth noting that A′ exhibited the highest variance of the five arms (std=0.0285), making it the noisiest-behaving arm alongside being the best-performing one.)*

## Phase 2: Thermal Transport (Experiment 3)

**Objective:**
Test whether the original "rejected structures improve representation learning" hypothesis holds on a completely disjoint, non-electronic property task: a phonon-related property (weighted last-phonon-DOS peak frequency, cm⁻¹) — a vibrational/lattice-dynamics quantity related to, but distinct from, thermal conductivity, on the `matbench_phonons` dataset.

**Methodology & Corrections:**
Following the discovery of the batch-dynamics confound in Phase 1, this experiment was designed from the ground up to be rigorously controlled. The A′ matched-step-count protocol was adopted as the baseline, meaning all arms took exactly the same number of gradient updates per epoch. To compensate for the very small size of `matbench_phonons` (1,265 structures total), the ratio of accepted to rejected data was increased to 1:1, and the experiment was run across 20 independent seeds to isolate variance. 

Additionally, a subtle data leakage flaw in Phase 1 (where model checkpoints were implicitly selected based on test-set performance) was corrected here. Convergence was monitored via a strict ~200-point internal validation split, with the 253-point test set reserved entirely blinded until the final epoch selection. 

**Results:**
- **Arm A′ (Baseline)**: Mean MAE = 128.35 ± 14.17
- **Arm B1 (Rejected, No Aux)**: Mean MAE = 125.10 ± 10.05
- **Arm B2 (Rejected + Aux)**: Mean MAE = 123.61 ± 11.43
- **Arm C (Rejected + Shuffled Aux)**: Mean MAE = 124.51 ± 13.16

The primary test (B2 vs A′) showed a mean relative improvement of +3.69% (exceeding the preregistered 3% threshold), but the 95% CI crossed zero asymmetrically ([-1.60, +11.07]), and the win/loss split across 20 seeds was a near coin-flip (11 wins, 9 losses), with $p=0.134$. 

Most decisively, the mutual-comparison matrix between the three augmented arms (B1, B2, C) yielded a clean statistical null. All pairwise comparisons among them showed $\le 1\%$ relative difference and $p > 0.50$. 

**Conclusion:**
The primary hypothesis test (B2 vs A′) is statistically **INCONCLUSIVE**. The point estimate improvement cannot be ruled out as noise due to the wide confidence intervals inherent in training on such a small dataset. However, the B1/B2/C matrix provides a decisively null outcome: the augmented arms are statistically indistinguishable from each other regardless of whether the auxiliary rejection-reason label is real, scrambled, or absent. 

Ultimately, whatever modest benefit might exist from introducing rejected data, it is cleanly unrelated to the auxiliary task's specific supervisory signal. The hypothesis that auxiliary structure-rejection learning fundamentally improves representations is **NOT SUPPORTED**, mirroring the corrected conclusions from Phase 1.
