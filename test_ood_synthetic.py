from download_model import download_if_missing
import pandas as pd
import numpy as np
import joblib
import sys
import random
import itertools
sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty, is_pure_metal
from miedema_test import read_miedema_db, calc_quaternary_hmix

def generate_synthetic_candidates(n=200):
    db = read_miedema_db('miedema_calc/database.dat')
    excluded = {'U', 'Pu', 'Th', 'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Tl', 'Hg', 'Cd', 'Os', 'Ir'}
    metals = [m for m in db.keys() if m not in excluded]
    
    passed_formulas = []
    metal_combos = list(itertools.combinations(metals, 4))
    
    # Keep going until we find `n` that pass Miedema
    random.seed(123)
    while len(passed_formulas) < n:
        combo = random.choice(metal_combos)
        splits = sorted(random.sample(range(1, 20), 3))
        fracs = (
            splits[0] * 0.05,
            (splits[1] - splits[0]) * 0.05,
            (splits[2] - splits[1]) * 0.05,
            (20 - splits[2]) * 0.05
        )
        
        hmix = calc_quaternary_hmix(db, combo, fracs)
        if hmix < 0:
            formula = f"{combo[0]}{fracs[0]:.2f}{combo[1]}{fracs[1]:.2f}{combo[2]}{fracs[2]:.2f}{combo[3]}{fracs[3]:.2f}"
            passed_formulas.append(formula)
            
    return passed_formulas

def print_stats(name, data):
    print(f"\n{name} Uncertainty Distribution:")
    print(f"  Min:    {np.min(data):.2f} K")
    print(f"  10th:   {np.percentile(data, 10):.2f} K")
    print(f"  25th:   {np.percentile(data, 25):.2f} K")
    print(f"  Median: {np.percentile(data, 50):.2f} K")
    print(f"  75th:   {np.percentile(data, 75):.2f} K")
    print(f"  90th:   {np.percentile(data, 90):.2f} K")
    print(f"  Max:    {np.max(data):.2f} K")

def main():
    print("Loading data and model...")
    df = pd.read_csv("data/supercon.csv")
    model = joblib.load(download_if_missing())
    
    real_metals = []
    
    # Get 200 real metallic compounds
    for _, row in df.iterrows():
        f = row['name']
        if pd.isna(f):
            continue
        try:
            feats = get_composition_features(f)
            if feats is None:
                continue
            if is_pure_metal(f):
                if len(real_metals) < 200:
                    real_metals.append((f, row['Tc'], feats))
            if len(real_metals) == 200:
                break
        except Exception:
            pass

    print("\n--- Real Metallic Alloys (Training Set) ---")
    X_real = np.array([m[2] for m in real_metals])
    uncerts_real = get_rf_uncertainty(model, X_real)
    print(f"Evaluated {len(real_metals)} real training compounds.")
        
    print("\nGenerating 200 Synthetic Miedema-passing Candidates...")
    synthetic_formulas = generate_synthetic_candidates(200)
    
    synthetic_candidates = []
    for f in synthetic_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            synthetic_candidates.append((f, feats))
            
    print("\n--- Synthetic Combinatorial Candidates ---")
    X_syn = np.array([c[1] for c in synthetic_candidates])
    uncerts_syn = get_rf_uncertainty(model, X_syn)
    print(f"Evaluated {len(synthetic_candidates)} synthetic compounds.")
        
    print_stats("Real Training Metals", uncerts_real)
    print_stats("Synthetic Metals", uncerts_syn)
    
    print("\nChecking threshold (90% training < T and 90% synthetic > T)...")
    p90_real = np.percentile(uncerts_real, 90)
    p10_syn = np.percentile(uncerts_syn, 10)
    print(f"90th percentile of real training: {p90_real:.2f} K")
    print(f"10th percentile of synthetic: {p10_syn:.2f} K")
    
    if p90_real < p10_syn:
        print("SUCCESS: A clean threshold exists.")
    else:
        print("FAILURE: No clean separation. The distributions overlap significantly.")

if __name__ == "__main__":
    main()
