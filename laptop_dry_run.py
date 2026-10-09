import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

print("--- LAPTOP DRY RUN (Stubbed Model & Mocks) ---")

def parse_id(x):
    return str(int(float(x)))

# Mocked idx_test_full.txt
idx_test = ['7.100000000000000000e+01', '0.000000000000000000e+00', '1.480000000000000000e+02', '2.970000000000000000e+02', '1.750000000000000000e+02']
test_ids = [parse_id(x) for x in idx_test]
port_check_ids = test_ids[:5]

run_ids = port_check_ids.copy()

print(f"Total run_ids: {len(run_ids)}")

# Mock structures
structures = {cid: "Atoms()" for cid in run_ids}

# Mock dataloader output (just use CSO predictions for the stub so Port Check passes!)
import urllib.request
cso_df = pd.read_json("https://raw.githubusercontent.com/henniggroup/BETE-NET/main/test_preds/CSO.json")
db_df = pd.read_json("https://raw.githubusercontent.com/henniggroup/BETE-NET/main/database.json")

df = pd.DataFrame([{"index": mpid, "target": np.zeros(51).tolist()} for mpid in run_ids])
df.set_index('index', inplace=True)
df['structure'] = [structures[i] for i in df.index]

# Stubbed model inference (we just populate pred_avg with CSO predictions directly for the 5 ids)
for i, tid in enumerate(port_check_ids):
    df.loc[tid, 'pred_avg'] = cso_df.loc[str(i), 'a2F']

print("=== PORT CHECK ===")
Freq_final = np.arange(0.25, 101, 2)
def local_get_target(row):
    y = np.interp(np.arange(0.25, 101, 0.1), row.Freq_meV, row.a2F)
    Y = np.interp(Freq_final, np.arange(0.25, 101, 0.1), savgol_filter(y, 101, 3, mode="interp"))
    return np.asarray([v if v > 0.0 else 0.0 for v in Y])

for i, tid in enumerate(port_check_ids):
    our_pred = np.array(df.loc[tid, 'pred_avg'])
    pub_a2f = np.array(cso_df.loc[str(i), 'a2F'])
    
    db_row = db_df.loc[int(tid)]
    dft_target = local_get_target(db_row)
    
    max_cso = np.max(np.abs(our_pred - pub_a2f))
    max_dft = np.max(np.abs(our_pred - dft_target))
    print(f"ID {tid}: max diff vs CSO.json={max_cso:.6e}, max diff vs DFT target={max_dft:.6e}")
    assert np.allclose(our_pred, pub_a2f, rtol=1e-3, atol=1e-5)

print("\n=== CALIBRATION (Stubbed subset) ===")
# Just a mock to show the output format
print("n=5, MAE=0.000, bias=0.000, Spearman=1.000")
print("Residual std (lambda < 0.4) = 0.000, (lambda >= 0.4) = 0.000")
print("Tc MAE (Tc < 5 K) = 0.000, (Tc >= 5 K) = 0.000")

print("\nDone.")
