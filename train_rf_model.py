import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition
from superconductor.features import get_node_features
from sklearn.ensemble import RandomForestRegressor

def get_composition_features(formula):
    try:
        comp = Composition(formula).fractional_composition
        features = []
        weights = []
        for el, amt in comp.items():
            class MockSite:
                def __init__(self, e):
                    self.specie = e
            features.append(get_node_features(MockSite(el)))
            weights.append(amt)
        features = np.array(features)
        weights = np.array(weights).reshape(-1, 1)
        mean_feat = np.sum(features * weights, axis=0)
        return mean_feat
    except Exception:
        return None

print("Loading data...")
df = pd.read_csv("data/supercon.csv")
valid_X, valid_Y = [], []
for idx, row in df.iterrows():
    formula = str(row['name'])
    tc = float(row['Tc'])
    feat = get_composition_features(formula)
    if feat is not None:
        valid_X.append(feat)
        valid_Y.append(tc)

X = np.array(valid_X)
Y = np.array(valid_Y)

print("Training Random Forest on full dataset (Deep Ensemble proxy)...")
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, Y)
joblib.dump(rf, "models/rf_tc_model.joblib")
print("Model saved to models/rf_tc_model.joblib")
