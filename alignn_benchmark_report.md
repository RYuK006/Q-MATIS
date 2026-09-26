# Milestone A5 Architecture Benchmark: CGCNN vs ALIGNN

## Overview
This benchmark evaluates ALIGNN against our baseline CGCNN.
Both models were trained using 3 random seeds (42, 123, 456), 50 epochs, identical node/edge features, and identical early stopping constraints.

## Tc Prediction Performance (3 Seeds, 50 Epochs)
| Metric | CGCNN (Mean ± Std) | ALIGNN (Mean ± Std) |
|---|---|---|
| MAE | 2.0350 ± 0.2013 | 1.9080 ± 0.1816 |
| RMSE | 7.2649 ± 3.2193 | 5.4397 ± 3.5804 |

## Loss Curves (Seed 42)

### CGCNN (Seed 42)
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | 42.6196 | 6.2803 |
| 5 | 40.5273 | 5.0606 |
| 10 | 39.0317 | 5.0569 |
| 15 | 36.0106 | 3.2666 |
| 20 | 32.2355 | 2.8392 |
| 25 | 26.2014 | 3.0353 |
| 30 | 13.8613 | 3.7537 |
| 35 | 4.8509 | 3.3034 |
| 40 | 6.4805 | 4.2901 |
| 41 | 4.8768 | 4.2375 |

### ALIGNN (Seed 42)
| Epoch | Train Loss | Val Loss |
|---|---|---|
| 0 | 43.1198 | 5.3429 |
| 5 | 40.6809 | 4.4811 |
| 10 | 37.2966 | 3.9154 |
| 15 | 32.7689 | 3.7317 |
| 20 | 29.8040 | 3.5469 |
| 25 | 20.5068 | 13.5023 |
| 30 | 19.3161 | 15.4140 |
| 35 | 13.6293 | 10.6566 |