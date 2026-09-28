from download_model import download_if_missing
import numpy as np
import joblib
import random
import time
import sys

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from score_phase1 import generate_phase1_candidates
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance
import itertools

def print_ratio_stats(name, data):
    print(f"\n{name} Relative Uncertainty Ratio (Uncertainty / Tc):")
    print(f"  Min:    {np.min(data):.4f}")
    print(f"  10th:   {np.percentile(data, 10):.4f}")
    print(f"  25th:   {np.percentile(data, 25):.4f}")
    print(f"  Median: {np.percentile(data, 50):.4f}")
    print(f"  75th:   {np.percentile(data, 75):.4f}")
    print(f"  90th:   {np.percentile(data, 90):.4f}")
    print(f"  Max:    {np.max(data):.4f}")

def main():
    model = joblib.load(download_if_missing())
    
    # 1. YBCO / Pnictides (Narrow Phase 1)
    print("Generating Narrow Phase 1 candidates...")
    narrow_candidates = generate_phase1_candidates()
    X_narrow = []
    narrow_formulas = []
    for f in narrow_candidates:
        feats = get_composition_features(f)
        if feats is not None:
            X_narrow.append(feats)
            narrow_formulas.append(f)
            
    X_narrow = np.array(X_narrow)
    preds_narrow = model.predict(X_narrow)
    uncerts_narrow = get_rf_uncertainty(model, X_narrow)
    
    # Sort narrow candidates
    narrow_results = []
    for i, f in enumerate(narrow_formulas):
        # Prevent division by zero
        tc = max(preds_narrow[i], 0.001)
        u = uncerts_narrow[i]
        ratio = u / tc
        narrow_results.append((f, tc, u, ratio))
        
    narrow_results.sort(key=lambda x: x[1], reverse=True)
    top_20_narrow = narrow_results[:20]
    
    # 2. General Phase 1 Sweep Sample
    print("\nGenerating General Phase 1 candidates (Sampling until ~6000 pass charge balance)...")
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    random.seed(42)
    general_formulas = []
    
    # Try to get ~6000
    while len(general_formulas) < 6000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            general_formulas.append(formula)
            
    X_general = []
    valid_general_formulas = []
    for f in general_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            X_general.append(feats)
            valid_general_formulas.append(f)
            
    X_general = np.array(X_general)
    preds_general = model.predict(X_general)
    uncerts_general = get_rf_uncertainty(model, X_general)
    
    general_results = []
    for i, f in enumerate(valid_general_formulas):
        tc = max(preds_general[i], 0.001)
        u = uncerts_general[i]
        ratio = u / tc
        general_results.append((f, tc, u, ratio))
        
    general_results.sort(key=lambda x: x[1], reverse=True)
    top_20_general = general_results[:20]
    
    # Extract ratios for full distribution
    ratios_narrow = [r[3] for r in narrow_results]
    ratios_general = [r[3] for r in general_results]
    
    print_ratio_stats("Narrow Phase 1 (Known Chemistry)", ratios_narrow)
    print_ratio_stats("General Phase 1 (Combinatorial Sweep)", ratios_general)
    
    cutoff = 0.40
    print(f"\nEvaluating CUTOFF: Ratio < {cutoff} (Uncertainty is less than 40% of predicted Tc)")
    
    narrow_survived = [r for r in top_20_narrow if r[3] < cutoff]
    general_survived = [r for r in top_20_general if r[3] < cutoff]
    
    print(f"\nTop 20 Narrow Phase 1 (Known Chemistry) survival rate: {len(narrow_survived)}/20")
    for f, tc, u, r in top_20_narrow:
        marker = "[SURVIVED]" if r < cutoff else "[EXCLUDED]"
        print(f"  {f:20} Tc: {tc:6.2f} ± {u:5.2f} K (Ratio: {r:.2f}) {marker}")
        
    print(f"\nTop 20 General Phase 1 (Combinatorial) survival rate: {len(general_survived)}/20")
    for f, tc, u, r in top_20_general:
        marker = "[SURVIVED]" if r < cutoff else "[EXCLUDED]"
        print(f"  {f:20} Tc: {tc:6.2f} ± {u:5.2f} K (Ratio: {r:.2f}) {marker}")

if __name__ == "__main__":
    main()
