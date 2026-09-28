import numpy as np
import pandas as pd
import joblib
import random
import time
import sys
import itertools
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def main():
    model = joblib.load("models/rf_tc_model.joblib")
    
    # 1. Gather Real SuperCon Data
    print("Loading SuperCon dataset...")
    df = pd.read_csv("data/supercon.csv")
    valid_formulas = df['name'].dropna().tolist()
    
    random.seed(100)
    random_supercon = random.sample(valid_formulas, 3000)
    
    X_real_feats = []
    real_forms = []
    for f in random_supercon:
        feats = get_composition_features(f)
        if feats is not None:
            X_real_feats.append(feats)
            real_forms.append(f)
    X_real_feats = np.array(X_real_feats)
    
    preds_real = model.predict(X_real_feats)
    uncerts_real = get_rf_uncertainty(model, X_real_feats)
    
    real_data = np.column_stack((preds_real, uncerts_real))
    real_labels = np.ones(len(real_data))
    
    # 2. Gather Combinatorial Noise Data
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    random.seed(200)
    gen_formulas = []
    
    while len(gen_formulas) < 3000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            gen_formulas.append(formula)
            
    X_gen_feats = []
    for f in gen_formulas:
        feats = get_composition_features(f)
        if feats is not None:
            X_gen_feats.append(feats)
    X_gen_feats = np.array(X_gen_feats)
    
    preds_gen = model.predict(X_gen_feats)
    uncerts_gen = get_rf_uncertainty(model, X_gen_feats)
    
    gen_data = np.column_stack((preds_gen, uncerts_gen))
    gen_labels = np.zeros(len(gen_data))
    
    # Create master lists
    all_formulas = real_forms + gen_formulas
    X_all = np.vstack((real_data, gen_data))
    y_all = np.concatenate((real_labels, gen_labels))
    
    # Split keeping track of indices so we can retrieve the formulas
    indices = np.arange(len(y_all))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, indices, test_size=0.3, random_state=42, stratify=y_all
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm.fit(X_train_scaled, y_train)
    
    preds_test = svm.predict(X_test_scaled)
    
    # Identify False Negatives (Real compounds predicted as combinatorial noise)
    false_negatives = []
    true_positives = []
    for i in range(len(y_test)):
        actual = y_test[i]
        predicted = preds_test[i]
        global_idx = idx_test[i]
        formula = all_formulas[global_idx]
        pred_tc = X_test[i][0]
        uncert = X_test[i][1]
        
        if actual == 1 and predicted == 0:
            false_negatives.append((formula, pred_tc, uncert))
        elif actual == 1 and predicted == 1:
            true_positives.append((formula, pred_tc, uncert))
            
    print(f"Total True Positives (Real preserved): {len(true_positives)}")
    print(f"Total False Negatives (Real excluded): {len(false_negatives)}")
    
    if len(false_negatives) > 0:
        avg_tc_tp = np.mean([x[1] for x in true_positives])
        avg_tc_fn = np.mean([x[1] for x in false_negatives])
        
        print(f"\nAverage Predicted Tc of True Positives:  {avg_tc_tp:.2f} K")
        print(f"Average Predicted Tc of False Negatives: {avg_tc_fn:.2f} K")
        
        print(f"\nSample of 30 False Negatives (Real compounds incorrectly excluded):")
        
        # Sort false negatives by Tc to see if they are clustered low
        false_negatives.sort(key=lambda x: x[1])
        
        # Take a spread
        sample_fns = false_negatives[:10] + false_negatives[len(false_negatives)//2:len(false_negatives)//2+10] + false_negatives[-10:]
        # Remove duplicates from simple slicing overlap
        sample_fns = list({v[0]:v for v in sample_fns}.values())
        sample_fns.sort(key=lambda x: x[1])
        
        for f, tc, u in sample_fns:
            print(f"  {f:25} Predicted Tc: {tc:6.2f} ± {u:5.2f} K")

if __name__ == "__main__":
    main()
