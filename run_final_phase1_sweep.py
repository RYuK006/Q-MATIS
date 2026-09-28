import numpy as np
import joblib
import itertools
from multiprocessing import Pool, cpu_count
import sys
import time

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def process_batch(formulas):
    valid_feats = []
    valid_forms = []
    for f in formulas:
        feats = get_composition_features(f)
        if feats is not None:
            valid_feats.append(feats)
            valid_forms.append(f)
    return valid_forms, valid_feats

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    print("Loading models...")
    rf_model = joblib.load("models/rf_tc_model.joblib")
    scaler, svm_model = joblib.load("models/ood_svm.joblib")
    
    print("Generating combinatorial space...")
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    
    metal_combos = list(itertools.combinations(metals, 3))
    total_combinations = len(metal_combos) * len(anions) * len(stoichs)
    print(f"Total theoretical space: {total_combinations:,} candidates")
    
    # 1. Filter by charge balance
    print("Applying charge balance filter...")
    t0 = time.time()
    charge_balanced_formulas = []
    
    for m in metal_combos:
        for anion in anions:
            for s in stoichs:
                if check_charge_balance((m[0], m[1], m[2], anion), s):
                    f = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
                    charge_balanced_formulas.append(f)
                    
    t1 = time.time()
    num_cb = len(charge_balanced_formulas)
    print(f"Charge balanced candidates: {num_cb:,} (took {t1-t0:.1f}s)")
    
    # 2. Featurize with Multiprocessing
    print(f"Featurizing {num_cb:,} candidates using {cpu_count()} cores...")
    chunk_size = max(1000, num_cb // (cpu_count() * 4))
    chunks = list(chunk_list(charge_balanced_formulas, chunk_size))
    
    t0 = time.time()
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_batch, chunks)
        
    all_forms = []
    all_feats = []
    for forms, feats in results:
        all_forms.extend(forms)
        all_feats.extend(feats)
        
    t1 = time.time()
    num_feat = len(all_forms)
    print(f"Successfully featurized: {num_feat:,} candidates (took {t1-t0:.1f}s)")
    
    X = np.array(all_feats)
    
    # 3. Predict Tc and Uncertainty
    print("Predicting Tc and estimating uncertainty...")
    t0 = time.time()
    preds = rf_model.predict(X)
    uncerts = get_rf_uncertainty(rf_model, X)
    t1 = time.time()
    print(f"Prediction took {t1-t0:.1f}s")
    
    # 4. Filter 1: Tc >= 20K
    print("Applying hard >20K filter...")
    high_tc_mask = preds >= 20.0
    
    high_tc_forms = np.array(all_forms)[high_tc_mask]
    high_tc_preds = preds[high_tc_mask]
    high_tc_uncerts = uncerts[high_tc_mask]
    
    print(f"Candidates with predicted Tc >= 20K: {len(high_tc_forms):,}")
    
    if len(high_tc_forms) == 0:
        print("No candidates above 20K.")
        return
        
    # 5. Filter 2: SVM OOD Filter
    print("Applying SVM OOD filter to high-Tc candidates...")
    X_svm = np.column_stack((high_tc_preds, high_tc_uncerts))
    X_svm_scaled = scaler.transform(X_svm)
    
    svm_preds = svm_model.predict(X_svm_scaled)
    
    # 1 = Real/In-Domain, 0 = Combinatorial/OOD
    passed_mask = svm_preds == 1
    
    final_forms = high_tc_forms[passed_mask]
    final_preds = high_tc_preds[passed_mask]
    final_uncerts = high_tc_uncerts[passed_mask]
    
    print(f"Candidates passing SVM filter: {len(final_forms):,}")
    
    # Sort and display Top 50
    results_list = list(zip(final_forms, final_preds, final_uncerts))
    results_list.sort(key=lambda x: x[1], reverse=True)
    
    print("\n--- FINAL PHASE 1 RESULTS: TOP 50 ---")
    for f, tc, u in results_list[:50]:
        print(f"{f:25} Predicted Tc: {tc:6.2f} ± {u:5.2f} K")

if __name__ == "__main__":
    main()
