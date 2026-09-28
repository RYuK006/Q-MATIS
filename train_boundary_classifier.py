import numpy as np
import pandas as pd
import joblib
import random
import time
import sys
import itertools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score
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
    # Using 3000 to keep featurization time reasonable
    random_supercon = random.sample(valid_formulas, 3000)
    
    print("Featurizing 3000 Real SuperCon candidates...")
    X_real_feats = []
    for f in random_supercon:
        feats = get_composition_features(f)
        if feats is not None:
            X_real_feats.append(feats)
    X_real_feats = np.array(X_real_feats)
    
    preds_real = model.predict(X_real_feats)
    uncerts_real = get_rf_uncertainty(model, X_real_feats)
    
    # Data structure: [predicted_tc, uncertainty]
    real_data = np.column_stack((preds_real, uncerts_real))
    real_labels = np.ones(len(real_data))
    
    print(f"Obtained {len(real_data)} real samples.")
    
    # 2. Gather Combinatorial Noise Data
    print("Generating ~3000 Combinatorial Phase 1 Candidates...")
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    metal_combos = list(itertools.combinations(metals, 3))
    
    random.seed(200)
    gen_formulas = []
    
    # Sample until we have 3000 valid
    while len(gen_formulas) < 3000:
        m = random.choice(metal_combos)
        anion = random.choice(anions)
        s = random.choice(stoichs)
        formula = f"{m[0]}{s[0]}{m[1]}{s[1]}{m[2]}{s[2]}{anion}{s[3]}"
        
        if check_charge_balance((m[0], m[1], m[2], anion), s):
            gen_formulas.append(formula)
            
    print("Featurizing 3000 Combinatorial candidates...")
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
    
    print(f"Obtained {len(gen_data)} combinatorial samples.")
    
    # 3. Combine and Split Data (70% train, 30% test)
    X_all = np.vstack((real_data, gen_data))
    y_all = np.concatenate((real_labels, gen_labels))
    
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42, stratify=y_all)
    
    print(f"\nData Split:")
    print(f"Training Set: {len(X_train)} samples ({sum(y_train==1)} Real, {sum(y_train==0)} Combinatorial)")
    print(f"Testing Set:  {len(X_test)} samples ({sum(y_test==1)} Real, {sum(y_test==0)} Combinatorial)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train Boundary Classifier (Logistic Regression & SVM)
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    
    print("Training SVM (RBF Kernel)...")
    svm = SVC(kernel='rbf', class_weight='balanced', probability=True)
    svm.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    def evaluate_model(name, clf, X_tr, y_tr, X_te, y_te):
        preds_tr = clf.predict(X_tr)
        preds_te = clf.predict(X_te)
        
        tr_prec = precision_score(y_tr, preds_tr)
        tr_rec = recall_score(y_tr, preds_tr)
        tr_f1 = f1_score(y_tr, preds_tr)
        
        te_prec = precision_score(y_te, preds_te)
        te_rec = recall_score(y_te, preds_te)
        te_f1 = f1_score(y_te, preds_te)
        
        print(f"\n--- {name} Results ---")
        print(f"Training Set:")
        print(f"  Precision: {tr_prec:.4f}")
        print(f"  Recall:    {tr_rec:.4f}")
        print(f"  F1-Score:  {tr_f1:.4f}")
        
        print(f"Test Set (Held-Out):")
        print(f"  Precision: {te_prec:.4f}")
        print(f"  Recall:    {te_rec:.4f}")
        print(f"  F1-Score:  {te_f1:.4f}")
        
        diff = tr_f1 - te_f1
        if diff > 0.05:
            print(f"  -> OVERFITTING DETECTED (F1 dropped by {diff:.4f} on Test)")
        elif diff < -0.05:
            print(f"  -> STRANGE: Test is significantly better than Train.")
        else:
            print(f"  -> STABLE: Generalization is solid. Delta F1 = {diff:.4f}")
            
    evaluate_model("Logistic Regression", lr, X_train_scaled, y_train, X_test_scaled, y_test)
    evaluate_model("Support Vector Machine (RBF)", svm, X_train_scaled, y_train, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
