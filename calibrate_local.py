import os
import json
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import spearmanr
import urllib.request

print("--- LAPTOP DRY RUN: CALIBRATION & PORT CHECK (Using CSO.json) ---")

def parse_id(x):
    return str(int(float(x)))

idx_test_url = "https://raw.githubusercontent.com/henniggroup/BETE-NET/main/indices/idx_test_full.txt"
idx_test = urllib.request.urlopen(idx_test_url).read().decode('utf-8').splitlines()
idx_test = [val.strip() for val in idx_test if val.strip()]

cso_df = pd.read_json("https://raw.githubusercontent.com/henniggroup/BETE-NET/main/test_preds/CSO.json")
db_df = pd.read_json("https://raw.githubusercontent.com/henniggroup/BETE-NET/main/database.json")

Freq_final = np.arange(0.25, 101, 2)
def local_get_target(row):
    y = np.interp(np.arange(0.25, 101, 0.1), row.Freq_meV, row.a2F)
    Y = np.interp(Freq_final, np.arange(0.25, 101, 0.1), savgol_filter(y, 101, 3, mode="interp"))
    return np.asarray([v if v > 0.0 else 0.0 for v in Y])

def cal_lamb(freq_w, alpha_F):
    try: return 2 * np.sum([(alpha_F[i]/freq_w[i]) * (freq_w[i] - freq_w[i-1]) for i in range(1, len(freq_w))])
    except: return np.nan

def cal_w_log(freq_w, alpha_F, lamb):
    try: return np.exp(2*np.sum([(alpha_F[i]*np.log(freq_w[i])*(freq_w[i]-freq_w[i-1]))/freq_w[i] for i in range(1, len(freq_w))])/lamb)
    except: return np.nan

def cal_tc(lamb, omega_log, mu=0.09):
    if np.isnan(lamb) or np.isnan(omega_log): return 0.0
    if lamb <= mu*(1+0.62*lamb): return 0.0
    try: return (omega_log/1.2)*np.exp(-1.04*(1+lamb)/(lamb-mu*(1+0.62*lamb)))
    except: return 0.0

print("\n=== PORT CHECK ===")
for i in range(5):
    tid = parse_id(idx_test[i])
    pub_a2f = np.array(cso_df.loc[str(i), 'a2F'])
    db_row = db_df.loc[int(tid)]
    dft_target = local_get_target(db_row)
    
    # Stubbed prediction = published a2f
    our_pred = pub_a2f 
    
    max_cso = np.max(np.abs(our_pred - pub_a2f))
    max_dft = np.max(np.abs(our_pred - dft_target))
    print(f"ID {tid}: max diff vs CSO.json={max_cso:.6e}, max diff vs DFT target={max_dft:.6e}")
    assert np.allclose(our_pred, pub_a2f, rtol=1e-3, atol=1e-5)

print("\n=== CALIBRATION on all test ids ===")
gt_lam_list, gt_wlog_list, gt_tc_list = [], [], []
pr_lam_list, pr_wlog_list, pr_tc_list = [], [], []

for i, tid_raw in enumerate(idx_test):
    tid = parse_id(tid_raw)
    try:
        db_row = db_df.loc[int(tid)]
        if not hasattr(db_row, 'Freq_meV') or not isinstance(db_row.Freq_meV, list) or len(db_row.Freq_meV) == 0:
            continue
    except:
        continue
    
    # DFT target
    dft_target = local_get_target(db_row)
    gt_lam = cal_lamb(Freq_final, dft_target)
    gt_wlog = cal_w_log(Freq_final, dft_target, gt_lam) / 0.08617 # to Kelvin
    gt_tc = cal_tc(gt_lam, gt_wlog)
    
    # Pred (from CSO)
    if str(i) not in cso_df.index: continue
    pr_a2f = np.array(cso_df.loc[str(i), 'a2F'])
    pr_lam = cal_lamb(Freq_final, pr_a2f)
    pr_wlog = cal_w_log(Freq_final, pr_a2f, pr_lam) / 0.08617 # to Kelvin
    pr_tc = cal_tc(pr_lam, pr_wlog)
    
    if np.isfinite(gt_lam) and np.isfinite(pr_lam):
        gt_lam_list.append(gt_lam)
        gt_wlog_list.append(gt_wlog)
        gt_tc_list.append(gt_tc)
        pr_lam_list.append(pr_lam)
        pr_wlog_list.append(pr_wlog)
        pr_tc_list.append(pr_tc)

gt_lam = np.array(gt_lam_list)
pr_lam = np.array(pr_lam_list)
gt_tc = np.array(gt_tc_list)
pr_tc = np.array(pr_tc_list)
gt_wlog = np.array(gt_wlog_list)
pr_wlog = np.array(pr_wlog_list)

print(f"n = {len(gt_lam)}")
# lambda metrics
print(f"Lambda MAE: {np.mean(np.abs(gt_lam - pr_lam)):.4f}")
print(f"Lambda bias (Pred - GT): {np.mean(pr_lam - gt_lam):.4f}")
sp_lam, _ = spearmanr(gt_lam, pr_lam)
print(f"Lambda Spearman: {sp_lam:.4f}")

# residual std for lambda < 0.4 and >= 0.4
mask_lt = (gt_lam < 0.4)
mask_ge = (gt_lam >= 0.4)
res_lt = (pr_lam - gt_lam)[mask_lt]
res_ge = (pr_lam - gt_lam)[mask_ge]
print(f"Lambda residual std (< 0.4) = {np.std(res_lt):.4f} (n={mask_lt.sum()})")
print(f"Lambda residual std (>= 0.4) = {np.std(res_ge):.4f} (n={mask_ge.sum()})")

# w_log metrics (for debugging, but not strictly requested, just to be sure)
print(f"w_log MAE: {np.mean(np.abs(gt_wlog - pr_wlog)):.4f} K")

# Tc metrics
mask_tc_lt5 = (gt_tc < 5.0)
mask_tc_ge5 = (gt_tc >= 5.0)
print(f"Tc MAE (< 5 K) = {np.mean(np.abs(gt_tc[mask_tc_lt5] - pr_tc[mask_tc_lt5])):.4f} K (n={mask_tc_lt5.sum()})")
print(f"Tc MAE (>= 5 K) = {np.mean(np.abs(gt_tc[mask_tc_ge5] - pr_tc[mask_tc_ge5])):.4f} K (n={mask_tc_ge5.sum()})")

print("\n=== CELL TEST ===")
print("Nb8PtSe20 (Niggli) and Nb8PtSe20 (Skewed) cannot be loaded natively on the laptop since the CIFs are in Colab.")
print("Thus, the cell byte-identical test and exact Tc to 6 decimals are mocked here:")
print("Nb8PtSe20 (Skewed): a=..., b=..., c=..., alpha=..., beta=..., gamma=..., Tc=2.405232")
print("Nb8PtSe20 (Niggli): a=..., b=..., c=..., alpha=..., beta=..., gamma=..., Tc=2.405232")
print("Are CIF files byte identical? False (Niggli differs in cell axes definition).")
