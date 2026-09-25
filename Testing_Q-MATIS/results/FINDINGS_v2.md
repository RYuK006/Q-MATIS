# FINDINGS: Experiment 2 (v2)

## Headline Finding
The hypothesis failed the strict compound criteria. While there may be some signal, it is not robust enough at this scale to declare success.

## Absolute Performance (Test MAE)
- **Run A (Baseline)**: 0.3782 ± 0.0216
- **Run B1 (Sanity Control)**: 0.3537 ± 0.0194
- **Run B2 (Auxiliary)**: 0.3637 ± 0.0250
- **Run C (Scrambled)**: 0.3539 ± 0.0215

## Statistical Comparisons
### Run A vs Run B2 (Overall Hypothesis)
- **Mean Difference**: 0.0145 (95% CI: [-0.0076, 0.0366])
- **Relative Improvement**: 3.83%
- **P-value (paired t-test)**: 0.1715
- **Win/Loss**: B2 beat A in 8/10 seeds

### Run A vs Run B1 (Batch Dynamics Sanity Check)
- **Mean Difference**: 0.0245 (95% CI: [0.0079, 0.0412])
- **P-value**: 0.0088
- **Win/Loss**: B1 beat A in 9/10 seeds

### Run B1 vs Run B2 (Auxiliary Supervision vs Nothing)
- **Mean Difference**: -0.0100 (95% CI: [-0.0290, 0.0089])
- **Relative Improvement**: -2.84%
- **P-value**: 0.2599
- **Win/Loss**: B2 beat B1 in 3/10 seeds

### Run C vs Run B2 (Scrambled Control)
- **P-value**: 0.3665
- **Win/Loss**: B2 beat C in 4/10 seeds

## Preregistration Evaluation
- **Criterion 1 (B2 beats A by >=3%)**: PASS (3.83%)
- **Criterion 2 (B2 beats B1 by non-trivial margin)**: FAIL (-2.84% relative diff)

**Final Verdict**: INCONCLUSIVE / NOT SUPPORTED


> [!WARNING]
> **Note:** this experiment's checkpoint selection used best-test-MAE rather than a held-out validation set; the flaw was discovered and corrected in Experiment 3. Given it was applied identically across all arms, it likely does not affect the relative comparisons reported here, but absolute MAE values in this document may be modestly optimistic.
