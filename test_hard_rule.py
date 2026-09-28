from download_model import download_if_missing
import pandas as pd
import numpy as np
import joblib
import random
import sys
import itertools
from sklearn.metrics import f1_score, precision_score, recall_score
import time

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def main():
    print("Loading models...")
    rf_model = joblib.load(download_if_missing())
    
    # 1. Real SuperCon > 20K
    print("Evaluating Real SuperCon data...")
    df = pd.read_csv("data/supercon.csv")
    valid_formulas = df['name'].dropna().tolist()
    
    # Extract all real SuperCon features
    X_real = []
    f_real = []
    for f in valid_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            X_real.append(feats)
            f_real.append(f)
    X_real = np.array(X_real)
    preds_real = rf_model.predict(X_real)
    uncert_real = get_rf_uncertainty(rf_model, X_real)
    
    # Filter > 20K
    idx_gt20 = np.where(preds_real > 20.0)[0]
    real_gt20_uncert = uncert_real[idx_gt20]
    
    # Split 50/50
    np.random.seed(42)
    shuffled_idx = np.random.permutation(len(idx_gt20))
    split = len(idx_gt20) // 2
    train_idx = idx_gt20[shuffled_idx[:split]]
    test_idx = idx_gt20[shuffled_idx[split:]]
    
    train_real_uncert = uncert_real[train_idx]
    test_real_uncert = uncert_real[test_idx]
    test_real_preds = preds_real[test_idx]
    
    print(f"Real SuperCon > 20K total: {len(idx_gt20)}")
    print(f"  Split 1 (Train): {len(train_real_uncert)}")
    print(f"  Split 2 (Test): {len(test_real_uncert)}")
    
    # 2. Training Noise > 20K
    print("Evaluating Training Noise data...")
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    random.seed(200)
    gen_formulas_train = []
    while len(gen_formulas_train) < 3000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            gen_formulas_train.append(formula)
            
    X_gen_train = []
    for f in gen_formulas_train:
        feats = get_composition_features(f)
        if feats is not None:
            X_gen_train.append(feats)
    X_gen_train = np.array(X_gen_train)
    preds_noise = rf_model.predict(X_gen_train)
    uncert_noise = get_rf_uncertainty(rf_model, X_gen_train)
    
    noise_gt20_uncert = uncert_noise[preds_noise > 20.0]
    print(f"Training Noise > 20K: {len(noise_gt20_uncert)}")
    
    # 3. Find optimal threshold using Split 1 and Training Noise > 20K
    # We want to separate real (class 1) from noise (class 0)
    y_true = np.concatenate([np.ones(len(train_real_uncert)), np.zeros(len(noise_gt20_uncert))])
    uncerts = np.concatenate([train_real_uncert, noise_gt20_uncert])
    
    best_thresh = 25.0
    best_f1 = -1
    for t in np.linspace(10, 40, 301):
        y_pred = (uncerts < t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            
    print(f"\nOptimization Result:")
    print(f"  Best Threshold: {best_thresh:.2f} K")
    print(f"  Best F1 Score: {best_f1:.4f}")
    
    final_thresh = best_thresh
    if abs(best_thresh - 25.0) > 3.0:
        print(f"  Using optimal threshold {best_thresh:.2f} instead of 25 because it significantly improves F1 score on the training split.")
    else:
        print(f"  Optimal threshold is near 25. Using {best_thresh:.2f}.")
        
    # 4. Fresh Pilot of 10,000 valid candidates
    print("\nGenerating fresh batch of 10,000 candidates (seed 12345)...")
    random.seed(12345)
    fresh_candidates = []
    while len(fresh_candidates) < 10000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            fresh_candidates.append(formula)
            
    X_pilot = []
    valid_f = []
    for f in fresh_candidates:
        feats = get_composition_features(f)
        if feats is not None:
            X_pilot.append(feats)
            valid_f.append(f)
    X_pilot = np.array(X_pilot)
    preds_pilot = rf_model.predict(X_pilot)
    uncert_pilot = get_rf_uncertainty(rf_model, X_pilot)
    
    # 5. Report Recall on Held-Out Real SuperCon
    test_real_passed = np.sum(test_real_uncert < final_thresh)
    recall = test_real_passed / len(test_real_uncert)
    print(f"\n--- TEST REPORT ---")
    print(f"Recall on Real SuperCon (Held-out half): {recall:.4f} ({test_real_passed}/{len(test_real_uncert)})")
    if recall < 0.80:
        print("WARNING: Recall on real compounds is below 80%.")
        
    # 6. Report survivors on Fresh Pilot
    survivors = []
    for i in range(len(valid_f)):
        if preds_pilot[i] > 20.0 and uncert_pilot[i] < final_thresh:
            survivors.append((valid_f[i], preds_pilot[i], uncert_pilot[i]))
            
    print(f"Fresh pilot candidates surviving (>20K & uncert < {final_thresh:.2f}): {len(survivors)}")
    
    survivors.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 Survivors:")
    ybco_lookalikes = 0
    for i in range(min(20, len(survivors))):
        f, tc, u = survivors[i]
        print(f"  {i+1:2}. {f:20} Tc: {tc:6.2f} ± {u:5.2f} K")
        if 'Cu' in f and 'O' in f and any(x in f for x in ['Ba', 'Sr', 'Ca', 'Y', 'La']):
            ybco_lookalikes += 1
            
    print(f"\nOut of top {min(20, len(survivors))}, Cuprate/YBCO lookalikes: {ybco_lookalikes}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Finished in {time.time()-t0:.1f}s")
