# Goal Description

The goal is to test the Q-MATIS "Scientific Memory" Hypothesis (Experiment 2). We will test whether training a materials property predictor (ALIGNN or CGCNN) on ACCEPTED + REJECTED candidates (with rejection reason encoded as an auxiliary signal) achieves lower test MAE on formation energy compared to training on ACCEPTED-only candidates. This requires a controlled setup with fixed architecture, fixed seed, and fixed effective dataset size.

## User Review Required

> [!WARNING]
> I will install the `alignn` and `matbench` python packages using `pip install alignn matbench` to support the experiment as ALIGNN is the preferred architecture and Matbench provides a clean self-contained dataset for formation energy.

## Open Questions

> [!IMPORTANT]
> If `alignn` fails to install or is too heavy for the current compute environment, I will fall back to CGCNN as requested. Do you approve this fallback?

## Proposed Changes

We will create a self-contained pipeline within the `c:/Users/Aaron/Desktop/Q-MATIS H1 001` directory.

### Project Structure

#### [NEW] results/preregistration.md
The preregistration document outlining the hypothesis and threshold, written before any training starts.

#### [NEW] results/dataset_provenance.md
A log of the dataset source (Matbench `matbench_mp_e_form` or MP API), query parameters, date pulled, and row count.

#### [NEW] experiments/exp2_memory_hypothesis/01_data_prep.py
A script to:
1. Download a subset of structures (e.g., 20,000 structures).
2. Generate a "rejected" set (~5000 structures) using structure perturbations (random atomic displacement/substitution) and a cheap physics filter (e.g., PyMatgen structural validity or simple bond distance checks).
3. Save accepted and rejected structures to disk (e.g. JSON or Parquet).

#### [NEW] experiments/exp2_memory_hypothesis/02_train.py
A unified training script that can run both Run A and Run B.
- **Run A:** Standard regression on accepted materials.
- **Run B:** Multi-task learning on accepted + rejected materials (formation energy MSE + lambda * rejection reason CrossEntropy).
- Implements seeded runs (up to 3 seeds depending on compute time).

#### [NEW] results/FINDINGS.md
Final writeup with metrics (MAE, RMSE), variance across seeds, and a direct verdict on whether Run B beat Run A by the 3% relative MAE threshold.

## Verification Plan

### Automated Tests
- I will execute the data prep script and verify the generated rejected dataset looks reasonable.
- I will run 1 epoch of Run A and Run B as a smoke test to ensure loss decreases and multi-task loss is properly computed.

### Manual Verification
- The final metrics and train/val curves will be recorded in `FINDINGS.md` for review.
