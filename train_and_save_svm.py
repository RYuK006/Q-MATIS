import numpy as np
import pandas as pd
import joblib
import random
import sys
import itertools
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

sys.path.append('.')
from run_comp_model import get_composition_features
from test_ood import get_rf_uncertainty
from general_phase1 import get_valid_metals, get_valid_anions, get_stoichiometries, check_charge_balance

def main():
    model = joblib.load("models/rf_tc_model.joblib")
    
    print("Gathering training data for Final SVM...")
    df = pd.read_csv("data/supercon.csv")
    valid_formulas = df['name'].dropna().tolist()
    
    random.seed(100)
    random_supercon = random.sample(valid_formulas, 3000)
    
    X_real_feats = []
    for f in random_supercon:
        feats = get_composition_features(f)
        if feats is not None:
            X_real_feats.append(feats)
    X_real_feats = np.array(X_real_feats)
    
    preds_real = model.predict(X_real_feats)
    uncerts_real = get_rf_uncertainty(model, X_real_feats)
    real_data = np.column_stack((preds_real, uncerts_real))
    real_labels = np.ones(len(real_data))
    
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
    
    X_all = np.vstack((real_data, gen_data))
    y_all = np.concatenate((real_labels, gen_labels))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    
    print("Fitting SVM...")
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_scaled, y_all)
    
    print("Saving to models/ood_svm.joblib")
    joblib.dump((scaler, svm), "models/ood_svm.joblib")
    print("Done.")

if __name__ == "__main__":
    main()
