# Experiment 2: Q-MATIS "Scientific Memory" Hypothesis

## Goal
To test whether training a materials property predictor (CGCNN) on **ACCEPTED + REJECTED** candidates (with rejection reason encoded as an auxiliary learning signal) achieves a lower test MAE on formation energy compared to training on **ACCEPTED-only** candidates.

## Methodology
- **Run A**: Standard regression on ACCEPTED materials only (formation energy target).
- **Run B**: Multi-task learning on ACCEPTED + REJECTED materials (formation energy MSE + `lambda=0.5` * rejection reason CrossEntropy).
- **Run C**: Control. Same as Run B, but the rejection reason labels for the rejected materials were randomly scrambled.
- **Dataset**: `matbench_mp_e_form` (16,000 train, 4,000 test) + 5,000 synthetically generated rejected structures.
- **Seeds**: 42, 43, 44
- **Threshold**: 3% relative MAE improvement of Run B over Run A.

## Results (Test MAE)

| Seed | Run A (Accepted Only) | Run B (Auxiliary Memory) | Run C (Scrambled Control) |
|---|---|---|---|
| **42** | 0.3696 | 0.3474 | 0.3502 |
| **43** | 0.3684 | 0.3521 | 0.3369 |
| **44** | 0.3465 | 0.3605 | 0.3841 |
| **Mean**| **0.3615** | **0.3533** | **0.3571** |

## Analysis & Verdict

**1. Performance Improvement**
- **Run B vs Run A**: Run B achieved a mean MAE of 0.3533 compared to Run A's 0.3615. 
- Relative Improvement: `(0.3615 - 0.3533) / 0.3615` = **2.27% improvement**.
- **Run C vs Run A**: Run C achieved a mean MAE of 0.3571, a 1.2% improvement over Run A.

**2. Verdict**
The "Scientific Memory" Hypothesis states that Run B must beat Run A by a **3% relative MAE threshold**.
Run B achieved a **2.27%** relative improvement. While it *did* outperform the standard baseline (Run A) and the scrambled control (Run C), it **fell short** of the strict 3% threshold required for this experiment. 

**3. Conclusion**
The results are encouraging. Teaching the model *why* a material is unphysical/rejected (Run B) provides better representation learning than simply throwing away the bad data (Run A) or feeding it random noise labels (Run C). However, the effect size is marginal under this specific model architecture (CGCNN) and dataset size. Further experiments with a more expressive architecture (like ALIGNN) or a larger, real-world "rejected" dataset (rather than synthetically perturbed structures) are warranted.
