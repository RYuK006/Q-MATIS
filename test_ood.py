import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition
import sys
sys.path.append('.')
from run_comp_model import get_composition_features

def get_rf_uncertainty(model, X):
    # For a RandomForestRegressor, predict for each tree and take the std dev
    preds = []
    for tree in model.estimators_:
        preds.append(tree.predict(X))
    return np.std(preds, axis=0)

def is_pure_metal(formula):
    try:
        comp = Composition(formula)
        for el in comp.elements:
            if not el.is_metal and not el.is_metalloid:
                return False
        return True
    except Exception:
        return False

def main():
    print("Loading data and model...")
    df = pd.read_csv("data/supercon.csv")
    model = joblib.load("models/rf_tc_model.joblib")
    
    metals = []
    ceramics = []
    
    for _, row in df.iterrows():
        f = row['name']
        if pd.isna(f):
            continue
        try:
            feats = get_composition_features(f)
            if feats is None:
                continue
            if is_pure_metal(f):
                if len(metals) < 20:
                    metals.append((f, row['Tc'], feats))
            else:
                if len(ceramics) < 20:
                    ceramics.append((f, row['Tc'], feats))
                    
            if len(metals) == 20 and len(ceramics) == 20:
                break
        except Exception:
            pass
            
    print("\n--- Pure Metallic Alloys ---")
    X_metals = np.array([m[2] for m in metals])
    preds_metals = model.predict(X_metals)
    uncerts_metals = get_rf_uncertainty(model, X_metals)
    for i, (f, true_tc, _) in enumerate(metals):
        print(f"{f:15} -> True: {true_tc:5.1f} | Pred: {preds_metals[i]:5.1f} ± {uncerts_metals[i]:5.2f} K")
        
    print("\n--- Ceramics / Pnictides ---")
    X_ceramics = np.array([c[2] for c in ceramics])
    preds_ceramics = model.predict(X_ceramics)
    uncerts_ceramics = get_rf_uncertainty(model, X_ceramics)
    for i, (f, true_tc, _) in enumerate(ceramics):
        print(f"{f:15} -> True: {true_tc:5.1f} | Pred: {preds_ceramics[i]:5.1f} ± {uncerts_ceramics[i]:5.2f} K")
        
    print(f"\nMean Uncertainty (Metals): {np.mean(uncerts_metals):.2f} K")
    print(f"Mean Uncertainty (Ceramics): {np.mean(uncerts_ceramics):.2f} K")

if __name__ == "__main__":
    main()
