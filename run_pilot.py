from download_model import download_if_missing
import itertools
import random
import time
import joblib
import numpy as np
import sys
from collections import Counter

sys.path.append('.')
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from sklearn.preprocessing import StandardScaler

def main():
    print("Loading models...")
    rf_model = joblib.load(download_if_missing())
    scaler, svm_model = joblib.load("models/ood_svm.joblib")
    
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    print(f"Sampling fresh from {len(metal_combos)*len(anions)*len(stoichs)} combinatorial space...")
    random.seed(999) # Fresh seed
    
    passed_charge_balance = []
    generated = 0
    t0 = time.time()
    
    # Generate ~10000 valid candidates
    while len(passed_charge_balance) < 10000:
        generated += 1
        m_combo = random.choice(metal_combos)
        anion = random.choice(anions)
        stoich = random.choice(stoichs)
        
        if check_charge_balance(m_combo + (anion,), stoich):
            formula_str = f"{m_combo[0]}{stoich[0]}{m_combo[1]}{stoich[1]}{m_combo[2]}{stoich[2]}{anion}{stoich[3]}"
            passed_charge_balance.append(formula_str)
            if len(passed_charge_balance) % 2000 == 0:
                print(f"Found {len(passed_charge_balance)} valid charge-balanced candidates (sampled {generated} total)")
                
    print(f"\nPILOT GENERATION REPORT:")
    print(f"  Total raw combinations generated/sampled: {generated}")
    print(f"  Number passing charge balance: {len(passed_charge_balance)}")
    
    print("\nFeaturizing and Scoring...")
    X_feats = []
    valid_formulas = []
    for f in passed_charge_balance:
        feats = get_composition_features(f)
        if feats is not None:
            X_feats.append(feats)
            valid_formulas.append(f)
            
    X_feats = np.array(X_feats)
    preds = rf_model.predict(X_feats)
    uncerts = get_rf_uncertainty(rf_model, X_feats)
    
    # Prepare data for SVM: [predicted_tc, uncertainty]
    svm_data = np.column_stack((preds, uncerts))
    svm_data_scaled = scaler.transform(svm_data)
    svm_preds = svm_model.predict(svm_data_scaled)
    
    passed_svm_count = 0
    high_conf_results = []
    
    for i, f in enumerate(valid_formulas):
        tc = preds[i]
        u = uncerts[i]
        svm_pass = (svm_preds[i] == 1)
        
        # Rule: candidates predicted above 20 K get the SVM filter
        # candidates below 20 K are excluded from high-confidence
        if tc > 20.0:
            if svm_pass:
                passed_svm_count += 1
                high_conf_results.append((f, tc, u))
        else:
            # tc <= 20 K are excluded
            pass
            
    print(f"  Number passing the >20K + SVM filter: {passed_svm_count}")
    
    high_conf_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 High-Confidence Candidates (Passed >20K + SVM):")
    for i in range(min(20, len(high_conf_results))):
        f, tc, u = high_conf_results[i]
        print(f"  {i+1:2}. {f:20} Tc: {tc:6.2f} ± {u:5.2f} K")
        
    # Check for non-cuprate / YBCO lookalikes
    ybco_lookalikes = 0
    non_cuprates = []
    
    for f, tc, u in high_conf_results:
        # Simple heuristic: if it contains Cu and O and Ba/Sr/Ca/Y/La
        if 'Cu' in f and 'O' in f and any(x in f for x in ['Ba', 'Sr', 'Ca', 'Y', 'La']):
            ybco_lookalikes += 1
        else:
            if len(non_cuprates) < 5:
                non_cuprates.append((f, tc, u))
                
    print(f"\nAnalysis of High-Confidence Set ({len(high_conf_results)} total):")
    print(f"  YBCO/Cuprate Lookalikes: {ybco_lookalikes}")
    print(f"  Non-Cuprate Examples: {len(high_conf_results) - ybco_lookalikes}")
    if non_cuprates:
        print("  Some Non-Cuprate Examples:")
        for f, tc, u in non_cuprates[:3]:
            print(f"    - {f:20} Tc: {tc:6.2f} ± {u:5.2f} K")

if __name__ == '__main__':
    main()
