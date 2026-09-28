from download_model import download_if_missing
import pandas as pd
import numpy as np
import joblib
import random
import sys
import itertools
import time

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def main():
    print("Loading models...")
    rf_model = joblib.load(download_if_missing())
    scaler, svm_model = joblib.load("models/ood_svm.joblib")
    
    # 1. Evaluate Real SuperCon
    print("Evaluating Real SuperCon data...")
    df = pd.read_csv("data/supercon.csv")
    valid_formulas = df['name'].dropna().tolist()
    
    # Evaluate 3000 to match training 
    random.seed(100)
    random_supercon = random.sample(valid_formulas, min(3000, len(valid_formulas)))
    X_real = []
    for f in random_supercon:
        feats = get_composition_features(f)
        if feats is not None:
            X_real.append(feats)
    X_real = np.array(X_real)
    preds_real = rf_model.predict(X_real)
    uncert_real = get_rf_uncertainty(rf_model, X_real)
    
    real_gt_20 = preds_real > 20.0
    num_real_gt_20 = np.sum(real_gt_20)
    real_gt_20_uncert_gt_30 = np.sum((preds_real > 20.0) & (uncert_real > 30.0))
    fraction = real_gt_20_uncert_gt_30 / num_real_gt_20 if num_real_gt_20 > 0 else 0
    print(f"\n--- PART 3: Real SuperCon Noise Signal ---")
    print(f"Real SuperCon > 20K: {num_real_gt_20}")
    print(f"Real SuperCon > 20K AND Uncert > 30K: {real_gt_20_uncert_gt_30}")
    print(f"Fraction: {fraction:.4f}")
    
    # 2. Evaluate Training Noise
    print("\nEvaluating SVM Training Noise...")
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
    preds_gen = rf_model.predict(X_gen_train)
    uncert_gen = get_rf_uncertainty(rf_model, X_gen_train)
    
    gen_gt_20 = preds_gen > 20.0
    num_gen_gt_20 = np.sum(gen_gt_20)
    print(f"\n--- PART 2: SVM Training Noise ---")
    print(f"Training Noise Samples total: 3000")
    print(f"Training Noise Samples predicted > 20 K: {num_gen_gt_20}")
    
    print("\nTabulated Predicted Tc vs Uncertainty (Averages):")
    print(f"Real SuperCon > 20 K (n={num_real_gt_20}): Mean Tc={np.mean(preds_real[real_gt_20]):.1f}K, Mean Uncert={np.mean(uncert_real[real_gt_20]):.1f}K")
    if num_gen_gt_20 > 0:
        print(f"Train Noise > 20 K (n={num_gen_gt_20}): Mean Tc={np.mean(preds_gen[gen_gt_20]):.1f}K, Mean Uncert={np.mean(uncert_gen[gen_gt_20]):.1f}K")
    
    # 3. Fresh Pilot Sample for Survivors and Failures
    print("\nGenerating exactly the same 10000 pilot candidates to find survivors & failures...")
    random.seed(999) 
    passed_charge_balance = []
    while len(passed_charge_balance) < 10000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            passed_charge_balance.append(formula)
            
    X_pilot = []
    valid_f = []
    for f in passed_charge_balance:
        feats = get_composition_features(f)
        if feats is not None:
            X_pilot.append(feats)
            valid_f.append(f)
    X_pilot = np.array(X_pilot)
    preds_pilot = rf_model.predict(X_pilot)
    uncert_pilot = get_rf_uncertainty(rf_model, X_pilot)
    
    svm_data = np.column_stack((preds_pilot, uncert_pilot))
    svm_data_scaled = scaler.transform(svm_data)
    svm_preds = svm_model.predict(svm_data_scaled)
    svm_decision = svm_model.decision_function(svm_data_scaled)
    
    survivors = []
    survivor_decisions = []
    failures = []
    failure_decisions = []
    
    for i in range(len(valid_f)):
        if preds_pilot[i] > 20.0:
            if svm_preds[i] == 1:
                survivors.append((preds_pilot[i], uncert_pilot[i]))
                survivor_decisions.append(svm_decision[i])
            else:
                failures.append((preds_pilot[i], uncert_pilot[i]))
                failure_decisions.append(svm_decision[i])
                
    random.seed(42)
    sample_failures = random.sample(failure_decisions, min(200, len(failure_decisions)))
    
    print(f"\n--- PART 1: Survivors vs Failures ---")
    print(f"Pilot > 20K candidates: {len(survivors) + len(failures)}")
    print(f"Survivors (Passed SVM): {len(survivors)}")
    print(f"Failures (Failed SVM): {len(failures)}")
    
    if len(survivors) > 0:
        survs = np.array(survivors)
        print(f"Survivors > 20 K (n={len(survivors)}): Mean Tc={np.mean(survs[:,0]):.1f}K, Mean Uncert={np.mean(survs[:,1]):.1f}K")
        
    print("\nDecision Function Distribution (Positive = Pass, Negative = Fail):")
    print("Survivors:")
    print(f"  Min: {np.min(survivor_decisions):.4f}, Max: {np.max(survivor_decisions):.4f}, Mean: {np.mean(survivor_decisions):.4f}")
    print("Random 200 Failures (>20K):")
    print(f"  Min: {np.min(sample_failures):.4f}, Max: {np.max(sample_failures):.4f}, Mean: {np.mean(sample_failures):.4f}")
    
    print("\nSurvivor Decision Percentiles:")
    for p in [0, 25, 50, 75, 100]:
        print(f"  {p}th: {np.percentile(survivor_decisions, p):.4f}")
        
    print("\nFailure Decision Percentiles (of the 200 sample):")
    for p in [0, 25, 50, 75, 100]:
        print(f"  {p}th: {np.percentile(sample_failures, p):.4f}")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Finished in {time.time()-t0:.1f}s")
