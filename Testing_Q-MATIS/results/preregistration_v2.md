# Preregistration: Experiment 2 (Corrected Replication)

## Superseding Prior Preregistration
This document supersedes the original 3-arm, 3-seed preregistration for the Q-MATIS "Scientific Memory" hypothesis. The original design was underpowered (n=3) and crucially confounded: Run B simultaneously altered the dataset quantity (adding rejected candidates) AND the learning objective (adding the auxiliary rejection-reason prediction task). It was thus impossible to attribute performance gains purely to the auxiliary supervision as intended.

## New Experimental Design
We implement a 4-arm design with 10 seeds (42 through 51) to achieve higher statistical power and explicitly isolate variables. 

- **Run A (Baseline)**: Accepted materials only. Evaluates standard regression on formation energy.
- **Run B1 (Null/Batch-Dynamics Sanity Control)**: Accepted + Rejected materials. Rejected materials are included in graph batches but are masked from contributing to the formation energy loss and receive zero auxiliary loss (yielding zero gradients). This arm does *not* isolate a "data quantity" effect (since the rejected data contributes no learning signal); rather, it is a sanity check isolating the pure effect of batch size dynamics and reduced accepted-graphs-per-batch. It is expected to be statistically indistinguishable from A.
- **Run B2 (Auxiliary supervision)**: Accepted + Rejected materials. Rejected materials contribute a CrossEntropy auxiliary loss (`lambda = 0.5`) on the rejection reason. This separates the auxiliary supervision from "nothing" (as verified by B1).
- **Run C (Scrambled control)**: Same as Run B2, but with rejection reason labels randomly shuffled, breaking the semantic pairing.

## Hypothesis & Compound Success Criterion
The "Scientific Memory" hypothesis posits that teaching a network *why* data was rejected improves its representation for the primary task. 

For the hypothesis to be considered **supported**, the following compound criteria must be strictly met:
1. **B2 must beat A by ≥ 3% relative Test MAE.**
2. **B2 must beat B1 by a non-trivial margin.** The auxiliary supervision effect (B2) must explain performance variance beyond mere batch-dynamics effects (B1). (Note: This experiment does *not* cleanly separate "more data" from "auxiliary supervision", but separates "auxiliary supervision" from "nothing".)

If B1 is indistinguishable from B2, then the hypothesis is **not supported**.

## Controls, Leakage, & Loss Dynamics
- **Train and test splits** are strictly deterministic by seed.
- **Rejected (synthetic) candidates** will be generated strictly from the train split. 
- A programmatic strict **leakage guard** will assert no parent structure ID from the rejected set appears in the held-out test split.
- **Loss Averaging (Learning Rate Confound Guard)**: In mixed batches (Run B1, B2, C), `loss_e` (formation energy MSE) is computed strictly on the slice of accepted samples (e.g. `e_pred[mask_acc]`). Because PyTorch `F.mse_loss(reduction='mean')` averages over the provided tensor size, the loss is divided by the number of *accepted samples in the batch* (~97), not the total batch size (128). This avoids artificially suppressing the learning rate in mixed-batch runs compared to Run A (where it is divided by 128). The effective gradient magnitude per accepted sample remains stable, ensuring changes in MAE are not an artifact of an accidentally altered learning rate.
