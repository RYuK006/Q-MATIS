import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

def evaluate(csv_path, model_name):
    print(f"\n{'='*40}\nEvaluating {model_name} from {csv_path}\n{'='*40}")
    try:
        df_pred = pd.read_csv(csv_path).set_index('material_id')
    except Exception as e:
        print(f"File not found or error: {e}")
        return
        
    print(f"Total relaxed: {len(df_pred)}")
    
    df_wbm = pd.read_csv("wbm-summary.csv.gz").set_index('material_id')
    df = df_pred.join(df_wbm[['uncorrected_energy', 'e_correction_per_atom_mp2020']], how='inner')
    
    e_total_pred = df['e_total']
    n_atoms = df['n_atoms']
    
    if model_name == "CHGNet":
        # CHGNet is trained on MP2020 corrected energies. 
        e_total_true = df['uncorrected_energy'] + df['e_correction_per_atom_mp2020'] * n_atoms
    else:
        # MACE-MP is trained on uncorrected VASP trajectories.
        e_total_true = df['uncorrected_energy']
        
    # Mathematical identity: E_hull_pred - E_hull_true = (E_total_pred - E_total_true) / n_atoms
    df['e_hull_pred'] = df['e_above_hull_true'] + (e_total_pred - e_total_true) / n_atoms
    
    mae = mean_absolute_error(df['e_above_hull_true'], df['e_hull_pred'])
    print(f"MAE of predicted E_hull: {mae*1000:.1f} meV/atom")
    
    for thresh in [0.0, 0.025]:
        print(f"\nThreshold: E_hull <= {thresh*1000:.0f} meV/atom")
        y_true = (df['e_above_hull_true'] <= thresh).astype(int)
        y_pred = (df['e_hull_pred'] <= thresh).astype(int)
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  F1 Score:  {f1:.3f}")
        
    print("\nAtom count distribution of this sample:")
    bins = [0, 4, 10, 20, 40, float('inf')]
    labels = ["<=4", "5-10", "11-20", "21-40", ">40"]
    print(pd.cut(df['n_atoms'], bins=bins, labels=labels).value_counts().sort_index())

evaluate("chgnet_results.csv", "CHGNet")
evaluate("mace_results.csv", "MACE-MP")
