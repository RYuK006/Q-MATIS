# Analysis v3 Output: Run A' (Matched Step Count)

## Absolute Performance (Test MAE)
- **Run A (Baseline, ~125 steps)**: 0.3782 ± 0.0216
- **Run B1 (Zero-grad Control, ~164 steps)**: 0.3537 ± 0.0194
- **Run A' (Matched Baseline, ~164 steps)**: 0.3525 ± 0.0285
- **Run B2 (Auxiliary)**: 0.3637 ± 0.0250
- **Run C (Scrambled)**: 0.3539 ± 0.0215

## Statistical Comparisons

### Run A' vs Run A (Did step count improve baseline?)
- **Mean Difference**: 0.0258 (95% CI: [0.0172, 0.0344])
- **P-value**: 0.0001
- **Win/Loss**: A' beat A in 10/10 seeds

### Run A' vs Run B1 (Hypothesis Test: Does step count explain the gap?)
- **Mean Difference**: 0.0012 (95% CI: [-0.0199, 0.0224])
- **P-value**: 0.8976
- **Win/Loss**: A' beat B1 in 5/10 seeds

### Run A' vs Run B2 
- **Mean Difference**: 0.0113 (95% CI: [-0.0162, 0.0388])
- **P-value**: 0.3774
- **Win/Loss**: A' beat B2 in 6/10 seeds

### Run A' vs Run C
- **Mean Difference**: 0.0014 (95% CI: [-0.0190, 0.0219])
- **P-value**: 0.8774
- **Win/Loss**: A' beat C in 4/10 seeds
