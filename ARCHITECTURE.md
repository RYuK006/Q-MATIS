# Q-MATIS Architecture Document

## Overview
Q-MATIS currently utilizes a **Composition-Only** architecture for predicting superconductor critical temperatures (Tc). The architecture is designed to bypass the limitations of structure-based models when dealing with fractionally doped compounds.

## Why Composition-Only?
Superconducting datasets (like SuperCon) are predominantly composed of continuously doped, fractional solid solutions (e.g., `Ba0.4K0.6Fe2As2`). 
Structure-based architectures (like ALIGNN or CGCNN) require exact 3D crystal structures, which are typically fetched from databases like Materials Project. However, these databases catalog stoichiometric integer crystals, resulting in two fatal errors:
1. **Massive Data Loss:** Exact formula matching drops ~90% of the dataset.
2. **Feature Blindness:** Mapping fractional formulas to un-doped host structures strips the doping fraction out of the graph features, rendering the models incapable of distinguishing between highly doped and lightly doped variants of the same host crystal.

To solve this, Q-MATIS completely replaces the structure-resolution step with a composition-only feature extractor that reads the fractional values directly from the formula.

## The Pipeline

### 1. Composition Parser
Formulas are parsed using `pymatgen.core.Composition(formula).fractional_composition`. This generates a precise mapping of elements to their fractional presence in the material.

### 2. Elemental Feature Extraction
For each element in the composition, 17 elemental properties are extracted using a Magpie-style featurizer. These include:
- Atomic Number (Z), Atomic Mass
- Electronegativity, Covalent/Atomic/Ionic Radii
- Electron Affinity, Ionization Energy
- Group, Row, Block (s, p, d, f)
- Valence, Molar Volume, Polarizability
- Mendeleev Number, Common Oxidation States

### 3. Fractional Aggregation
The elemental features are combined into a single feature vector representing the entire composition. This is done via a stoichiometric weighted average:
```python
mean_feat = sum(element_feature_vector * fractional_amount)
```
This ensures the model inherently "sees" the exact doping levels.

### 4. Machine Learning Model (Random Forest)
The aggregated feature vectors are fed into a Random Forest Regressor (100 trees). This ensemble architecture natively handles non-linear relationships between the elemental features and the critical temperature.

### 5. Uncertainty Calibration (Deep Ensemble Proxy)
Uncertainty estimation is critical for materials discovery. Q-MATIS leverages the variance across the individual decision trees within the Random Forest to act as a proxy for Deep Ensembles.
- **Mean Tc:** The average prediction across all 100 trees.
- **Uncertainty (1σ):** The standard deviation of the predictions across all 100 trees.
- **95% Confidence Bounds:** Calculated using `Mean ± 1.96σ`.
