# Milestone A5 Architecture Benchmark: CGCNN vs ALIGNN

## Overview
This benchmark evaluates the state-of-the-art ALIGNN (Atomistic Line Graph Neural Network) against our baseline CGCNN.
Both models were trained using exactly the same data split (seed=42), identical node/edge features, and identical early stopping constraints.

## Tc Prediction Performance
| Metric | CGCNN | ALIGNN | Improvement |
|---|---|---|---|
| MAE | 1.8412 | 1.7884 | 2.87% |
| RMSE | 2.8999 | 2.8601 | 1.37% |
| R2 | 0.0662 | 0.0916 | - |

## Computational Profile
| Metric | CGCNN | ALIGNN |
|---|---|---|
| Train Time (s) | 12.48 | 9.89 |
| Inference (samples/s) | 2891.62 | 3031.11 |
| Peak GPU Mem (MB) | 62.03 | 75.27 |
| Parameters | 350,849 | 353,409 |

