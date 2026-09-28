from download_model import download_if_missing
import pandas as pd
import numpy as np
import joblib
import sys
sys.path.append('.')
from run_comp_model import get_composition_features

def main():
    print("Loading model...")
    model = joblib.load(download_if_missing())
    
    # Base: Ba(1-x) K(x) Fe2 As2
    fractions = [0.30, 0.35, 0.40, 0.45, 0.50]
    
    print("\n--- Doping-Fraction Sensitivity Check ---")
    print("Compound series: Ba(1-x) K(x) Fe2 As2")
    
    for x in fractions:
        ba_amt = 1.0 - x
        formula = f"Ba{ba_amt:.2f}K{x:.2f}Fe2As2"
        feats = get_composition_features(formula)
        if feats is not None:
            pred_tc = model.predict([feats])[0]
            print(f"{formula:20} -> Predicted Tc = {pred_tc:.2f} K")
        else:
            print(f"{formula:20} -> Feature generation failed")

if __name__ == "__main__":
    main()
