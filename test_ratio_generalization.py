from download_model import download_if_missing
import numpy as np
import joblib
import random
import time
import pandas as pd
import sys

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance
import itertools

def main():
    model = joblib.load(download_if_missing())
    
    # 1. 20 Random SuperCon Compounds (Fresh, NOT sorted by Tc)
    print("Loading SuperCon dataset and sampling 20 random compounds...")
    df = pd.read_csv("data/supercon.csv")
    valid_formulas = df['name'].dropna().tolist()
    
    # Remove obvious YBCO matches just to be safe
    filtered_formulas = [f for f in valid_formulas if not ("Y" in f and "Ba" in f and "Cu" in f)]
    
    random.seed(999) # different seed
    random_supercon = random.sample(filtered_formulas, 50)
    
    X_real = []
    real_formulas = []
    
    for f in random_supercon:
        try:
            feats = get_composition_features(f)
            if feats is not None:
                X_real.append(feats)
                real_formulas.append(f)
            if len(real_formulas) == 20:
                break
        except:
            pass
            
    X_real = np.array(X_real)
    preds_real = model.predict(X_real)
    uncerts_real = get_rf_uncertainty(model, X_real)
    
    real_results = []
    for i, f in enumerate(real_formulas):
        tc = max(preds_real[i], 0.001)
        u = uncerts_real[i]
        ratio = u / tc
        real_results.append((f, tc, u, ratio))
        
    # 2. 20 Fresh Combinatorial Hallucinations (Different seed)
    print("Generating 20 fresh combinatorial candidates...")
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    random.seed(888) # fresh seed
    general_formulas = []
    
    while len(general_formulas) < 20:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            general_formulas.append(formula)
            
    X_gen = []
    gen_formulas = []
    for f in general_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            X_gen.append(feats)
            gen_formulas.append(f)
            
    X_gen = np.array(X_gen)
    preds_gen = model.predict(X_gen)
    uncerts_gen = get_rf_uncertainty(model, X_gen)
    
    gen_results = []
    for i, f in enumerate(gen_formulas):
        tc = max(preds_gen[i], 0.001)
        u = uncerts_gen[i]
        ratio = u / tc
        gen_results.append((f, tc, u, ratio))
        
    cutoff = 0.40
    print(f"\nEvaluating CUTOFF: Ratio < {cutoff} (Uncertainty is less than 40% of predicted Tc)")
    
    real_survived = [r for r in real_results if r[3] < cutoff]
    gen_survived = [r for r in gen_results if r[3] < cutoff]
    
    print(f"\n20 Random Real SuperCon Compounds (Cross-Section) Survival: {len(real_survived)}/20")
    for f, tc, u, r in real_results:
        marker = "[SURVIVED]" if r < cutoff else "[EXCLUDED]"
        print(f"  {f:20} Tc: {tc:6.2f} ± {u:5.2f} K (Ratio: {r:.2f}) {marker}")
        
    print(f"\n20 Random Fresh Combinatorial Candidates Survival: {len(gen_survived)}/20")
    for f, tc, u, r in gen_results:
        marker = "[SURVIVED]" if r < cutoff else "[EXCLUDED]"
        print(f"  {f:20} Tc: {tc:6.2f} ± {u:5.2f} K (Ratio: {r:.2f}) {marker}")

if __name__ == "__main__":
    main()
