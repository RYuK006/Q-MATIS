import sys, os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pymatgen.core import Composition
from superconductor.features import get_node_features

# Build Magpie-style weighted features
def get_composition_features(formula):
    try:
        comp = Composition(formula).fractional_composition
        features = []
        weights = []
        for el, amt in comp.items():
            # Mock site with specie=el
            class MockSite:
                def __init__(self, e):
                    self.specie = e
            features.append(get_node_features(MockSite(el)))
            weights.append(amt)
            
        features = np.array(features)
        weights = np.array(weights).reshape(-1, 1)
        
        # Weighted mean
        mean_feat = np.sum(features * weights, axis=0)
        
        return mean_feat
    except Exception:
        return None

df = pd.read_csv("data/supercon.csv")
print(f"Total SuperCon rows: {len(df)}")

valid_X = []
valid_Y = []

for idx, row in df.iterrows():
    formula = str(row['name'])
    tc = float(row['Tc'])
    feat = get_composition_features(formula)
    if feat is not None:
        valid_X.append(feat)
        valid_Y.append(tc)

X = np.array(valid_X)
Y = np.array(valid_Y)
print(f"Successfully featurized {len(X)} compositions.")

# Train a simple MLP as a quick check
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"RF MAE: {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RF R2: {r2_score(y_test, y_pred):.4f}")
