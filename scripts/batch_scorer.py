import argparse
import os
import sqlite3
import pandas as pd
import numpy as np
import joblib
from pymatgen.core import Composition
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
from superconductor.features import get_node_features
import warnings
warnings.filterwarnings("ignore")

def get_composition_features(formula):
    try:
        comp = Composition(formula).fractional_composition
        features = []
        weights = []
        for el, amt in comp.items():
            class MockSite:
                def __init__(self, e):
                    self.specie = e
            features.append(get_node_features(MockSite(el)))
            weights.append(amt)
        features = np.array(features)
        weights = np.array(weights).reshape(-1, 1)
        mean_feat = np.sum(features * weights, axis=0)
        return mean_feat
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="Batch Candidate Scorer")
    parser.add_argument("--input", type=str, default="data/work_queue.csv", help="Input CSV file with 'formula' column")
    parser.add_argument("--output", type=str, default="results/candidates.db", help="Output SQLite database")
    parser.add_argument("--model", type=str, default="models/rf_tc_model.joblib", help="Path to trained joblib model")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    print(f"Loading model from {args.model}...")
    try:
        rf = joblib.load(args.model)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading input formulas from {args.input}...")
    df = pd.read_csv(args.input)
    if 'formula' not in df.columns:
        print("Error: Input CSV must have a 'formula' column.")
        return

    # Set up SQLite
    conn = sqlite3.connect(args.output)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            formula TEXT PRIMARY KEY,
            predicted_tc REAL,
            uncertainty REAL
        )
    ''')
    conn.commit()

    print("Scoring candidates...")
    batch_size = 1000
    formulas = df['formula'].tolist()
    
    for i in tqdm(range(0, len(formulas), batch_size)):
        batch = formulas[i:i+batch_size]
        
        X_batch = []
        valid_formulas = []
        for f in batch:
            feat = get_composition_features(f)
            if feat is not None:
                X_batch.append(feat)
                valid_formulas.append(f)
                
        if not X_batch:
            continue
            
        X_batch = np.array(X_batch)
        
        # Get individual tree predictions for uncertainty
        tree_preds = []
        for tree in rf.estimators_:
            tree_preds.append(tree.predict(X_batch))
            
        tree_preds = np.array(tree_preds)  # shape: (n_estimators, n_samples)
        
        mean_tc = np.mean(tree_preds, axis=0)
        std_tc = np.std(tree_preds, axis=0)
        
        # Insert into DB
        rows = list(zip(valid_formulas, mean_tc.tolist(), std_tc.tolist()))
        cursor.executemany('''
            INSERT OR REPLACE INTO predictions (formula, predicted_tc, uncertainty)
            VALUES (?, ?, ?)
        ''', rows)
        conn.commit()
        
    conn.close()
    print(f"Done! Results written to {args.output}")

if __name__ == "__main__":
    main()
