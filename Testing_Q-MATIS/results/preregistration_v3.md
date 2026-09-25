# Preregistration v3: The Matched-Step-Count Control

This is a follow-up control experiment to `preregistration_v2.md`. 

## Background
Experiment 2 (v2) found that Run B1 (a zero-gradient control) improved over baseline Run A with a statistically significant $p=0.009$. The mechanism behind this improvement is currently unconfirmed. 

Two leading hypotheses emerged:
1. **Step Count (Batch Dynamics):** Mixing rejected structures into training batches increased the number of gradient steps per epoch on accepted-material data. Specifically, B1/B2/C had ~164 steps per epoch, compared to Run A's ~125 steps per epoch. This alone, not any information content in the rejected structures, might explain B1's improvement.
2. **Loss-Averaging Denominator:** If `loss_e` in the mixed-batch arms was averaged over the full batch size instead of only the accepted samples, the mixed-batch arms would effectively receive a smaller gradient scale per step.

## Audit of Loss-Averaging Convention
Before proceeding, an explicit code audit was performed to resolve the loss-averaging denominator hypothesis. 

In `experiments/exp2_memory_hypothesis_v2/02_train.py` (lines 154-157), the energy loss is computed as follows:
```python
if mask_acc.sum() > 0:
    loss_e = F.mse_loss(e_pred[mask_acc].view(-1), batch.y[mask_acc].view(-1))
```
Since `F.mse_loss` uses `reduction='mean'` by default and is applied exclusively to the subset `e_pred[mask_acc]`, the loss is averaged **ONLY over the accepted samples present in the batch**, not over the full batch of 128. This means the gradient scale per accepted sample is **not diminished**. 

**Conclusion of Audit:** The loss-averaging denominator hypothesis is rejected as a possible mechanism. We now solely focus on the step count hypothesis.

## Hypothesis
Run A's improvement, when matched to B1/B2/C's step count via reduced batch size, will account for some or all of the original A-vs-B1 gap.

## Experimental Design
- **New Arm (Run A′):** Identical to the original Run A (accepted-only, zero auxiliary loss) but with the batch size reduced to match B1/B2/C's step count.
- **Batch Size Calculation:** Run B1 processed 21,000 structures at batch size 128 (~164 steps/epoch). Run A has 16,000 accepted structures. To achieve ~164 steps/epoch, the batch size for Run A′ is set to 98 (16,000 / 98 = 163.2 steps/epoch).
- **Seeds:** The same 10 seeds as before: 42–51.
- **All other hyperparameters:** (learning rate, optimizer, epochs, etc.) remain identical to the original Run A.

## Success Criterion
- If Run A′'s mean MAE is **statistically indistinguishable** from B1's mean MAE (overlapping confidence intervals, non-significant paired t-test), the step-count/batch-dynamics explanation is **CONFIRMED**. 
- If Run A′ still **underperforms B1 significantly**, step count alone does NOT explain the gap, and another uninvestigated factor becomes the next thing to check.
