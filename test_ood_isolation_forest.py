import pandas as pd
import numpy as np
import sys
import random
from sklearn.ensemble import IsolationForest
import itertools

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import is_pure_metal
from miedema_test import read_miedema_db, calc_quaternary_hmix

def generate_synthetic_candidates(n=200):
    db = read_miedema_db('miedema_calc/database.dat')
    excluded = {'U', 'Pu', 'Th', 'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Tl', 'Hg', 'Cd', 'Os', 'Ir'}
    metals = [m for m in db.keys() if m not in excluded]
    
    passed_formulas = []
    metal_combos = list(itertools.combinations(metals, 4))
    
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
    print(f"\n{name} Anomaly Score Distribution (Higher = More Anomalous):")
    print(f"  Min:    {np.min(data):.4f}")
    print(f"  10th:   {np.percentile(data, 10):.4f}")
    print(f"  25th:   {np.percentile(data, 25):.4f}")
    print(f"  Median: {np.percentile(data, 50):.4f}")
    print(f"  75th:   {np.percentile(data, 75):.4f}")
    print(f"  90th:   {np.percentile(data, 90):.4f}")
    print(f"  Max:    {np.max(data):.4f}")

def main():
    print("Loading SuperCon dataset and building Isolation Forest...")
    df = pd.read_csv("data/supercon.csv")
    
    all_features = []
    real_metals_feats = []
    
    for f in df['name'].dropna():
        try:
            feats = get_composition_features(f)
            if feats is not None:
                all_features.append(feats)
                if len(real_metals_feats) < 200 and is_pure_metal(f):
                    real_metals_feats.append(feats)
        except Exception:
            pass
            
    X_train = np.array(all_features)
    print(f"Training Isolation Forest on {X_train.shape[0]} SuperCon samples...")
    
    iso = IsolationForest(n_estimators=200, contamination=0.1, random_state=42)
    iso.fit(X_train)
    
    X_real = np.array(real_metals_feats)
    # Invert decision function: lower decision_function = more anomalous. We want higher = more anomalous.
    scores_real = -iso.decision_function(X_real)
    print(f"Evaluated {len(X_real)} real training metallic compounds.")
    
    print("\nGenerating 200 Synthetic Miedema-passing Candidates...")
    synthetic_formulas = generate_synthetic_candidates(200)
    
    synthetic_feats = []
    for f in synthetic_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            synthetic_feats.append(feats)
            
    X_syn = np.array(synthetic_feats)
    scores_syn = -iso.decision_function(X_syn)
    print(f"Evaluated {len(X_syn)} synthetic compounds.")
    
    print_stats("Real Training Metals", scores_real)
    print_stats("Synthetic Metals", scores_syn)
    
    print("\nChecking threshold (90% training < T and 90% synthetic > T)...")
    p90_real = np.percentile(scores_real, 90)
    p10_syn = np.percentile(scores_syn, 10)
    print(f"90th percentile of real training anomaly score: {p90_real:.4f}")
    print(f"10th percentile of synthetic anomaly score: {p10_syn:.4f}")
    
    if p90_real < p10_syn:
        print("SUCCESS: A clean threshold exists. Isolation Forest cleanly separates the sets.")
    else:
        print("FAILURE: No clean separation. The distributions still overlap significantly.")

if __name__ == "__main__":
    main()
