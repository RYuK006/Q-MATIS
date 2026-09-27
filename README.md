# Q-MATIS: An Experimental Platform for AI-Driven Materials Discovery

Q-MATIS (Quantitative Materials AI Training & Inference System) is a research codebase focused on predicting material properties using machine learning, specifically targeting superconducting critical temperatures (Tc).

## Current Status: Composition-Only Modeling

The project currently focuses on **composition-only modeling** for superconductor Tc prediction. 

Early iterations of this project attempted to train structure-based Graph Neural Networks (CGCNN, ALIGNN) on the SuperCon dataset by mapping chemical formulas to 3D crystal structures using the Materials Project database. However, rigorous data auditing revealed a fatal flaw in this approach:

1. **The Exact-Match Bottleneck:** SuperCon consists largely of continuously doped fractional solid solutions (e.g., `Ba0.4K0.6Fe2As2`), while the Materials Project catalogs stoichiometric integer crystals (e.g., `KBa2(FeAs)6`). Exact formula matching resulted in a massive ~90% data loss, shrinking a 16,414-sample dataset to just ~1,500 samples.
2. **The Doping Featurization Failure:** Mapping doped formulas to undoped host structures (e.g., mapping `Ba0.4K0.6Fe2As2` to the `BaFe2As2` crystal graph) meant the model literally could not see the doping fraction. `Ba0.4K0.6Fe2As2` and `Ba0.1K0.9Fe2As2` were rendered as identical graphs, turning the labels into contradictory noise. This led to severe overfitting and an R² of ~0.09.

**The Fix:** We abandoned structure-based models in favor of a composition-only proxy that can natively parse fractional stoichiometries. 
- By using Magpie-style elemental features weighted by stoichiometric fraction, we bypassed the Materials Project bottleneck and recovered the entire dataset (16,406 valid compositions).
- Training a Random Forest Regressor on these features yielded an **R² of 0.80** and an **MAE of 5.94 K**, confirming that the model effectively generalizes from the doping fractions.

## Features

- **Composition Featurization:** Extracts 17 physical/chemical properties per element (atomic mass, electronegativity, valence, etc.) and computes fractional weighted averages.
- **Deep Ensemble Proxy Uncertainty:** Uses the variance across individual trees in the Random Forest to provide 95% Confidence Intervals for Tc predictions.
- **CLI Demo:** A simple interactive script to predict Tc for any valid chemical formula.

## Demo Usage

To predict the critical temperature (Tc) of a composition along with its uncertainty bound:

```bash
python demo.py "Ba0.4K0.6Fe2As2"
```

**Example Output:**
```
==================================================
Q-MATIS Tc Prediction Demo
==================================================
Formula:          Ba0.4K0.6Fe2As2
Predicted Tc:     28.67 K
Uncertainty (1 std dev): +/- 5.25 K
95% Conf. Bounds: [18.38 K, 38.96 K]
==================================================
```

## Repository Structure

```
Q-MATIS/
├── data/               # Datasets (SuperCon)
├── models/             # Saved model weights (.joblib)
├── superconductor/     # Core library
│   ├── features.py     # Magpie-style feature extraction
│   └── ...
├── demo.py             # CLI prediction tool
└── train_rf_model.py   # Training script for the composition model
```

## Installation

```bash
git clone https://github.com/RYuK006/Q-MATIS.git
cd Q-MATIS
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```
