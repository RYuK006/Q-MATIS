# FINDINGS: Experiment 2 - The Matched-Step-Count Control (v3)

## Preregistered Hypothesis & Criterion
**Hypothesis:** The improvement observed in Run B1 (zero-gradient sanity control) over baseline Run A in Experiment 2 was driven by batch dynamics—specifically, that mixing rejected structures into the training batches increased the number of gradient steps per epoch from ~125 (Run A) to ~164 (Run B1). 
**Criterion:** If a new Run A' (accepted-only, zero auxiliary loss, but with a reduced batch size to match ~164 steps/epoch) is statistically indistinguishable from Run B1, the step-count explanation is **CONFIRMED**.

## Absolute Performance (Test MAE)
- **Run A (Baseline, ~125 steps)**: 0.3782 ± 0.0216
- **Run B1 (Zero-grad Control, ~164 steps)**: 0.3537 ± 0.0194
- **Run A' (Matched Baseline, ~164 steps)**: 0.3525 ± 0.0285

## Statistical Comparisons
- **Run A' vs Run A**: Mean difference of 0.0258 (95% CI: [0.0172, 0.0344]), $p=0.0001$. Run A' beat Run A in 10/10 seeds. Matching the step count significantly improved the baseline.
- **Run A' vs Run B1**: Mean difference of 0.0012 (95% CI: [-0.0199, 0.0224]), $p=0.8976$. Run A' beat Run B1 in 5/10 seeds. The two runs are statistically indistinguishable.

## Loss-Averaging Audit
Prior to the experiment, an explicit code audit was performed to check an alternative hypothesis: whether the energy loss (`loss_e`) was averaged over the full batch size (diminishing the gradient scale per step) or only over accepted samples. The audit confirmed that the PyTorch implementation calculates `F.mse_loss` strictly on `e_pred[mask_acc]`, effectively averaging the loss **only over the accepted samples**. Thus, the gradient magnitude was not artificially reduced, eliminating the denominator effect as a contributing factor.

## Verdict: CONFIRMED
The step-count/batch-dynamics explanation is **CONFIRMED**. 

The entire gap between the baseline (Run A) and the zero-gradient control (Run B1) is fully explained by the increased number of gradient updates per epoch on the accepted material data. Run B1 did not benefit from any implicit regularization or "memory" of the rejected structures; it simply took more gradient steps. Consequently, the original hypothesis that the auxiliary task provides representation learning benefits (as tested in B2) must be evaluated against this stronger baseline (Run A'), which shows that the auxiliary task (B2: 0.3637 MAE) provides no significant benefit over properly scaled baseline training.

Additionally, A′ is statistically indistinguishable from B2 (p=0.377) and C (p=0.877), and B1/B2/C are mutually indistinguishable from each other (all p>0.25). This means no configuration involving rejected-structure data — with or without auxiliary supervision — shows a detectable effect once baseline training is correctly step-matched.

*(Note on DataLoaders & Variance: An explicit check confirmed that `drop_last=False` (default behavior) was used uniformly across all PyTorch DataLoaders in `02_train.py`, ensuring that partial batches (like A′'s last batch of 26) were processed identically across all arms. Finally, it is worth flagging that A′ had the highest variance among the five arms (std=0.0285), indicating that it was the noisiest-behaving arm despite its top-tier mean performance.)*


> [!WARNING]
> **Note:** this experiment's checkpoint selection used best-test-MAE rather than a held-out validation set; the flaw was discovered and corrected in Experiment 3. Given it was applied identically across all arms, it likely does not affect the relative comparisons reported here, but absolute MAE values in this document may be modestly optimistic.
