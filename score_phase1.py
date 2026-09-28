from download_model import download_if_missing
import numpy as np
import joblib
import pandas as pd
import sys

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty

def generate_phase1_candidates():
    candidates = set()
    
    # 1. Cuprate-like space (Y-Ba-Cu-O variations)
    # Y(1-x) Nd(x) Ba2 Cu3 O(7-d)
    rare_earths = ['Y', 'Nd', 'La', 'Sm', 'Gd']
    for re in rare_earths:
        for x in np.arange(0.0, 1.1, 0.1):
            for d in np.arange(0.0, 0.5, 0.1):
                y_amt = round(1.0 - x, 2)
                nd_amt = round(x, 2)
                o_amt = round(7.0 - d, 2)
                
                parts = []
                if y_amt > 0: parts.append(f"Y{y_amt}")
                if nd_amt > 0: parts.append(f"{re}{nd_amt}")
                parts.append("Ba2")
                parts.append("Cu3")
                parts.append(f"O{o_amt}")
                
                # if RE is Y, we might get Y twice. Let's fix that.
                if re == 'Y':
                    total_y = round(y_amt + nd_amt, 2)
                    formula = f"Y{total_y}Ba2Cu3O{o_amt}"
                    candidates.add(formula)
                else:
                    candidates.add("".join(parts))

    # 2. Iron-pnictide space (Ba-K-Fe-As variations)
    # Ba(1-x) K(x) Fe2 As2
    dopants = ['K', 'Na', 'Rb', 'Cs']
    for dopant in dopants:
        for x in np.arange(0.0, 1.1, 0.05):
            ba_amt = round(1.0 - x, 2)
            d_amt = round(x, 2)
            parts = []
            if ba_amt > 0: parts.append(f"Ba{ba_amt}")
            if d_amt > 0: parts.append(f"{dopant}{d_amt}")
            parts.append("Fe2As2")
            candidates.add("".join(parts))

    return list(candidates)

def main():
    print("Generating Phase 1 (ceramic/pnictide) candidates...")
    candidates = generate_phase1_candidates()
    print(f"Generated {len(candidates)} unique candidates.")
    
    print("Loading model...")
    model = joblib.load(download_if_missing())
    
    results = []
    
    # Featurize and score
    valid_candidates = []
    X = []
    
    for f in candidates:
        feats = get_composition_features(f)
        if feats is not None:
            valid_candidates.append(f)
            X.append(feats)
            
    X = np.array(X)
    print(f"Successfully featurized {len(valid_candidates)} candidates.")
    
    preds = model.predict(X)
    uncerts = get_rf_uncertainty(model, X)
    
    for i, f in enumerate(valid_candidates):
        results.append((f, preds[i], uncerts[i]))
        
    # Sort by predicted Tc descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Top 20 Phase 1 Candidates ---")
    for i in range(20):
        f, tc, u = results[i]
        print(f"{i+1:2}. {f:20} -> Predicted Tc = {tc:6.2f} ± {u:5.2f} K")
        
if __name__ == "__main__":
    main()
