from download_model import download_if_missing
import pandas as pd
import numpy as np
import joblib
import random
import sys
import itertools
import time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from pymatgen.core import Composition

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def get_element_group(formula):
    try:
        elements = [str(e) for e in Composition(formula).elements]
        return "-".join(sorted(elements))
    except:
        return "Unknown"

def main():
    print("Loading original RF model...")
    rf_model_original = joblib.load(download_if_missing())
    
    # 1. Evaluate original training noise (to find the 122)
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
    preds_noise = rf_model_original.predict(X_gen_train)
    uncert_noise = get_rf_uncertainty(rf_model_original, X_gen_train)
    
    # Keep only those > 20K
    noise_gt20_mask = preds_noise > 20.0
    uncert_noise_gt20 = uncert_noise[noise_gt20_mask]
    print(f"Training Noise > 20K: {len(uncert_noise_gt20)}")
    
    # 2. Fresh Pilot of 10,000 valid candidates
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
    for f in fresh_candidates:
        feats = get_composition_features(f)
        if feats is not None:
            X_pilot.append(feats)
    X_pilot = np.array(X_pilot)
    preds_pilot = rf_model_original.predict(X_pilot)
    uncert_pilot = get_rf_uncertainty(rf_model_original, X_pilot)
    
    pilot_gt20_mask = preds_pilot > 20.0
    uncert_pilot_gt20 = uncert_pilot[pilot_gt20_mask]
    print(f"Fresh pilot > 20K: {len(uncert_pilot_gt20)} (out of 10000)")
    
    # 3. SuperCon OOF Predictions with GroupKFold
    print("\nLoading Real SuperCon data and computing GroupKFold OOF predictions...")
    df = pd.read_csv("data/supercon.csv")
    df = df.dropna(subset=['name', 'Tc'])
    valid_formulas = df['name'].tolist()
    y_real = df['Tc'].values
    
    X_real = []
    y_valid = []
    groups = []
    for f, y in zip(valid_formulas, y_real):
        feats = get_composition_features(f)
        if feats is not None:
            X_real.append(feats)
            y_valid.append(y)
            groups.append(get_element_group(f))
            
    X_real = np.array(X_real)
    y_valid = np.array(y_valid)
    groups = np.array(groups)
    
    oof_preds = np.zeros(len(X_real))
    oof_uncert = np.zeros(len(X_real))
    
    gkf = GroupKFold(n_splits=5)
    fold = 1
    for train_idx, test_idx in gkf.split(X_real, y_valid, groups):
        print(f"  Training Fold {fold}...")
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_real[train_idx], y_valid[train_idx])
        oof_preds[test_idx] = rf.predict(X_real[test_idx])
        oof_uncert[test_idx] = get_rf_uncertainty(rf, X_real[test_idx])
        fold += 1
        
    real_gt20_mask = oof_preds > 20.0
    uncert_real_gt20 = oof_uncert[real_gt20_mask]
    num_real_gt20 = len(uncert_real_gt20)
    print(f"Real SuperCon (OOF) > 20K: {num_real_gt20}")
    
    # 4. Report stats for Thresholds
    thresholds = [10.0, 15.0, 20.0, 25.0, 30.0, 38.4]
    
    print("\n--- FINAL DIAGNOSTIC: THRESHOLD PERFORMANCE ---")
    print("Thresh | Recall (Real OOF >20K) | Noise Passthrough | Fresh Pilot Passthrough")
    print("-" * 75)
    for t in thresholds:
        # (a) recall on out-of-fold real compounds
        real_pass = np.sum(uncert_real_gt20 < t)
        recall = real_pass / num_real_gt20 if num_real_gt20 > 0 else 0
        
        # (b) fraction of the 122 training-noise samples that pass
        noise_pass = np.sum(uncert_noise_gt20 < t)
        noise_frac = noise_pass / len(uncert_noise_gt20) if len(uncert_noise_gt20) > 0 else 0
        
        # (c) fraction of 10,000 fresh pilot candidates that pass
        pilot_pass = np.sum(uncert_pilot_gt20 < t)
        pilot_frac = pilot_pass / 10000.0  # fraction of the 10,000 total candidates
        
        print(f"{t:6.1f} | {recall:20.4f} | {noise_frac:17.4f} | {pilot_frac:23.4f}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Finished in {time.time()-t0:.1f}s")
