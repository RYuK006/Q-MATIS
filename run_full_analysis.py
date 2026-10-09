import json
import re
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Add BETE-NET to path to import get_target
sys.path.append('BETE-NET-zip/BETE-NET-main')
from utils.data import get_target, Freq_final

# Helper class for get_target
class StructEntry:
    def __init__(self, Freq_meV, a2F):
        self.Freq_meV = Freq_meV
        self.a2F = a2F

def cal_lamb(freq_w, alpha_F):
    lambdaF = 0
    try:
        for i in range(1, len(freq_w)):
            dw = freq_w[i] - freq_w[i-1]
            w = freq_w[i]
            alpha_F_w = alpha_F[i]
            lambdaF += ((alpha_F_w / w) * dw)
        return 2 * lambdaF
    except:
        return np.nan

def cal_w_log(freq_w, alpha_F, lamb):
    w_logF = 0
    try:
        for i in range(1, len(freq_w)):
            dw = freq_w[i] - freq_w[i-1]
            w_logF += (alpha_F[i] * np.log(freq_w[i]) * dw / freq_w[i])
        return np.exp(2 * w_logF / lamb)
    except:
        return np.nan

def cal_tc_guarded(lamb, omega_log, mu=0.09):
    denom = lamb - mu * (1.0 + 0.62 * lamb)
    if denom <= 0 or np.isnan(lamb) or np.isnan(omega_log) or lamb <= 0 or omega_log <= 0:
        return 0.0
    frac = -1.04 * (1.0 + lamb) / denom
    tc = (omega_log / 1.2) * np.exp(frac)
    return tc if np.isfinite(tc) else 0.0

def main():
    print("=" * 80)
    print("BETE-NET 856-STRUCTURE RUN: FULL SCALE EVALUATION & CALIBRATION")
    print("=" * 80)

    # -------------------------------------------------------------
    # STEP 1: PARSE colab_run_856.txt
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("1. PARSE colab_run_856.txt")
    print("=" * 50)

    with open('colab_run_856.txt') as f:
        colab_raw = f.readlines()

    test_preds = {}   # tid -> (tc_str, tc_val, lam_val, wlog_val)
    mp_preds = {}     # mpid -> (tc_str, tc_val, lam_val, wlog_val)
    ext_preds = {}    # name -> (mpid, tc_val, lam_val, wlog_val)

    for line in colab_raw:
        line = line.strip()
        m_ext = re.match(r'^(.*?)\s*\((mp-[a-z0-9_]+)\):\s*Tc=([^,]+),\s*lambda=([^,]+),\s*w_log=([^\s]+)\s*K', line)
        if m_ext:
            name, mpid, tc_s, lam_s, wlog_s = m_ext.groups()
            tc_v = float(tc_s.replace('K', '').strip())
            ext_preds[name] = (mpid, tc_v, float(lam_s), float(wlog_s))
            continue
        
        m_line = re.match(r'^([^:]+):\s*Tc=([^,]+),\s*lambda=([^,]+),\s*w_log=([^\s]+)\s*K', line)
        if m_line:
            id_s, tc_s, lam_s, wlog_s = m_line.groups()
            id_s = id_s.strip()
            tc_clean = tc_s.replace('K', '').strip()
            tc_v = float(tc_clean) if tc_clean != 'inf' else np.inf
            lam_v = float(lam_s.strip())
            wlog_v = float(wlog_s.strip())
            if id_s.startswith('mp-'):
                mp_preds[id_s] = (tc_clean, tc_v, lam_v, wlog_v)
            elif id_s.isdigit():
                test_preds[id_s] = (tc_clean, tc_v, lam_v, wlog_v)

    print(f"Total test set lines parsed: {len(test_preds)}")
    print(f"Total mp- lines parsed in candidate block: {len(mp_preds)}")
    print(f"Total external reference lines parsed: {len(ext_preds)}")
    print(f"Total unique MP IDs across candidate and reference blocks: {len(set(mp_preds.keys()) | {v[0] for v in ext_preds.values()})}")

    # Load idx_test_full.txt
    with open('BETE-NET-zip/BETE-NET-main/indices/idx_test_full.txt') as f:
        idx_test_raw = [l.strip() for l in f if l.strip()]
    idx_test_ids = [str(int(float(x))) for x in idx_test_raw]

    missing_test_ids = [tid for tid in idx_test_ids if tid not in test_preds]
    print(f"\nidx_test_full.txt total IDs: {len(idx_test_ids)}")
    print(f"Matched test IDs in output: {len(idx_test_ids) - len(missing_test_ids)}")
    print(f"Missing test IDs from output ({len(missing_test_ids)}): {missing_test_ids}")

    # Load phonon stability results (43 candidates from report / 44 in CSV)
    df_ph = pd.read_csv('phonon_stability_results.csv')
    stable_cand_ids = df_ph[df_ph['status'] == 'STABLE']['material_id'].tolist()
    all_seen_mp = set(mp_preds.keys()) | {v[0] for v in ext_preds.values()}
    missing_stable_cands = [cid for cid in stable_cand_ids if cid not in all_seen_mp]
    print(f"\nTotal STABLE candidate IDs in CSV: {len(stable_cand_ids)}")
    print(f"Missing STABLE candidate IDs from printed output ({len(missing_stable_cands)}): {missing_stable_cands}")

    # -------------------------------------------------------------
    # STEP 2: PORT CHECK AT SCALE
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. PORT CHECK AT SCALE")
    print("=" * 50)

    with open('BETE-NET-zip/BETE-NET-main/test_preds/CSO.json') as f:
        cso = json.load(f)

    # Port check across the matched held-out test IDs (n = 170)
    matched_test_ids = [tid for tid in idx_test_ids if tid in test_preds and tid in cso['lamb_pred']]
    diff_l_test = []
    diff_w_test = []
    worst_test_records = []

    for tid in matched_test_ids:
        c_tc, c_tcv, c_l, c_w = test_preds[tid]
        # Published CSO predictions
        p_l = cso['lamb_pred'][tid]
        p_w = cso['wlog_pred'][tid]
        dl = abs(c_l - p_l)
        dw = abs(c_w - p_w)
        diff_l_test.append(dl)
        diff_w_test.append(dw)
        worst_test_records.append((tid, dl, dw, c_l, p_l, c_w, p_w))

    diff_l_test = np.array(diff_l_test)
    diff_w_test = np.array(diff_w_test)

    print(f"\n--- PORT CHECK: HELD-OUT TEST SET (n = {len(matched_test_ids)}) ---")
    print(f"n: {len(matched_test_ids)}")
    print(f"lambda: mean |dlambda| = {np.mean(diff_l_test):.6f}, max |dlambda| = {np.max(diff_l_test):.6f}")
    print(f"w_log:  mean |dw_log|  = {np.mean(diff_w_test):.6f} K, max |dw_log|  = {np.max(diff_w_test):.6f} K")
    fail_l_test = (diff_l_test > 2e-3).sum()
    fail_w_test = (diff_w_test > 0.5).sum()
    fail_either_test = ((diff_l_test > 2e-3) | (diff_w_test > 0.5)).sum()
    print(f"Count |dlambda| > 2e-3: {fail_l_test} / {len(matched_test_ids)} ({fail_l_test/len(matched_test_ids)*100:.1f}%)")
    print(f"Count |dw_log|  > 0.5 K: {fail_w_test} / {len(matched_test_ids)} ({fail_w_test/len(matched_test_ids)*100:.1f}%)")
    print(f"Count either > tol:    {fail_either_test} / {len(matched_test_ids)} ({fail_either_test/len(matched_test_ids)*100:.1f}%)")
    pass_pct_test = (1 - fail_either_test / len(matched_test_ids)) * 100
    print(f"Pass rate within tolerance: {pass_pct_test:.2f}% (Threshold: >= 99%) -> {'PASS' if pass_pct_test >= 99 else 'FAIL'}")

    print("\nWorst 5 IDs by |dlambda| (Held-out Test Set):")
    worst_by_l = sorted(worst_test_records, key=lambda x: x[1], reverse=True)[:5]
    for r in worst_by_l:
        print(f"  ID {r[0]}: |dlambda| = {r[1]:.4f} (pred: {r[3]:.3f}, cso: {r[4]:.4f}), |dw_log| = {r[2]:.3f} K")

    print("\nWorst 5 IDs by |dw_log| (Held-out Test Set):")
    worst_by_w = sorted(worst_test_records, key=lambda x: x[2], reverse=True)[:5]
    for r in worst_by_w:
        print(f"  ID {r[0]}: |dw_log| = {r[2]:.3f} K (pred: {r[5]:.3f}, cso: {r[6]:.3f}), |dlambda| = {r[1]:.4f}")

    # Port check across ALL 806 structures
    matched_all_ids = [tid for tid in test_preds if tid in cso['lamb_pred']]
    diff_l_all = []
    diff_w_all = []
    worst_all_records = []
    for tid in matched_all_ids:
        c_tc, c_tcv, c_l, c_w = test_preds[tid]
        p_l = cso['lamb_pred'][tid]
        p_w = cso['wlog_pred'][tid]
        dl = abs(c_l - p_l)
        dw = abs(c_w - p_w)
        diff_l_all.append(dl)
        diff_w_all.append(dw)
        worst_all_records.append((tid, dl, dw, c_l, p_l, c_w, p_w))

    diff_l_all = np.array(diff_l_all)
    diff_w_all = np.array(diff_w_all)
    print(f"\n--- PORT CHECK: FULL DATABASE (n = {len(matched_all_ids)}) ---")
    print(f"n: {len(matched_all_ids)}")
    print(f"lambda: mean |dlambda| = {np.mean(diff_l_all):.6f}, max |dlambda| = {np.max(diff_l_all):.6f}")
    print(f"w_log:  mean |dw_log|  = {np.mean(diff_w_all):.6f} K, max |dw_log|  = {np.max(diff_w_all):.6f} K")
    fail_l_all = (diff_l_all > 2e-3).sum()
    fail_w_all = (diff_w_all > 0.5).sum()
    fail_either_all = ((diff_l_all > 2e-3) | (diff_w_all > 0.5)).sum()
    print(f"Count |dlambda| > 2e-3: {fail_l_all} / {len(matched_all_ids)} ({fail_l_all/len(matched_all_ids)*100:.1f}%)")
    print(f"Count |dw_log|  > 0.5 K: {fail_w_all} / {len(matched_all_ids)} ({fail_w_all/len(matched_all_ids)*100:.1f}%)")
    print(f"Count either > tol:    {fail_either_all} / {len(matched_all_ids)} ({fail_either_all/len(matched_all_ids)*100:.1f}%)")
    pass_pct_all = (1 - fail_either_all / len(matched_all_ids)) * 100
    print(f"Pass rate within tolerance: {pass_pct_all:.2f}% (Threshold: >= 99%) -> {'PASS' if pass_pct_all >= 99 else 'FAIL'}")

    print("\nWorst 5 IDs by |dlambda| (Full 806 Set):")
    worst_all_by_l = sorted(worst_all_records, key=lambda x: x[1], reverse=True)[:5]
    for r in worst_all_by_l:
        print(f"  ID {r[0]}: |dlambda| = {r[1]:.4f} (pred: {r[3]:.3f}, cso: {r[4]:.4f}), |dw_log| = {r[2]:.3f} K")

    # -------------------------------------------------------------
    # STEP 3: CALIBRATION
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. CALIBRATION (database.json & get_target)")
    print("=" * 50)

    with open('BETE-NET-zip/BETE-NET-main/database.json') as f:
        db = json.load(f)

    # Check key type in database.json
    db_keys = list(db['comp'].keys())
    print(f"database.json key type: {type(db_keys[0])} (example: '{db_keys[0]}')")

    # We evaluate calibration on the held-out test set (n = 170) AND the full dataset (n = 806)
    def run_calibration(subset_ids, name):
        records = []
        for tid in subset_ids:
            if tid not in test_preds or tid not in db['comp']:
                continue
            c_tc_s, c_tcv, pred_l, pred_w = test_preds[tid]
            
            # Stored DFT from database.json
            dft_l_stored = db['lambda'][tid]
            # stored w_log in database.json is in meV -> convert to K
            dft_w_stored = db['w_log'][tid] / 0.08617

            # Recomputed DFT via get_target
            # Fast check: cso['lamb_target'] and cso['wlog_target'] were precomputed by author via get_target
            # Let's verify with actual get_target for this row
            row_obj = StructEntry(db['Freq_meV'][tid], db['a2F'][tid])
            tar = get_target(row_obj)
            dft_l_recomp = cal_lamb(Freq_final, tar)
            dft_w_recomp = cal_w_log(Freq_final, tar, dft_l_recomp) / 0.08617

            # Recomputed guarded Tc
            tc_pred_guard = cal_tc_guarded(pred_l, pred_w)
            tc_dft_stored_guard = cal_tc_guarded(dft_l_stored, dft_w_stored)
            tc_dft_recomp_guard = cal_tc_guarded(dft_l_recomp, dft_w_recomp)

            records.append({
                'id': tid,
                'comp': db['comp'][tid],
                'pred_l': pred_l,
                'pred_w': pred_w,
                'tc_pred_guard': tc_pred_guard,
                'dft_l_stored': dft_l_stored,
                'dft_w_stored': dft_w_stored,
                'tc_dft_stored_guard': tc_dft_stored_guard,
                'dft_l_recomp': dft_l_recomp,
                'dft_w_recomp': dft_w_recomp,
                'tc_dft_recomp_guard': tc_dft_recomp_guard
            })

        df_cal = pd.DataFrame(records)
        print(f"\n--- CALIBRATION RESULTS: {name} (n = {len(df_cal)}) ---")

        for dft_type in ['stored', 'recomp']:
            dft_l_col = f'dft_l_{dft_type}'
            dft_w_col = f'dft_w_{dft_type}'
            dft_tc_col = f'tc_dft_{dft_type}_guard'
            dft_label = "STORED DFT (database.json)" if dft_type == 'stored' else "RECOMPUTED DFT (get_target)"
            print(f"\n>> Target: {dft_label}")

            for regime, condition in [
                ("All lambda", df_cal[dft_l_col].notnull()),
                ("lambda < 0.4", df_cal[dft_l_col] < 0.4),
                ("lambda >= 0.4", df_cal[dft_l_col] >= 0.4)
            ]:
                sub = df_cal[condition]
                n = len(sub)
                if n == 0:
                    continue
                # Lambda stats
                err_l = sub['pred_l'] - sub[dft_l_col]
                mae_l = np.abs(err_l).mean()
                bias_l = err_l.mean()
                res_std_l = err_l.std(ddof=1)
                spear_l, _ = spearmanr(sub['pred_l'], sub[dft_l_col])

                # w_log stats
                err_w = sub['pred_w'] - sub[dft_w_col]
                mae_w = np.abs(err_w).mean()
                bias_w = err_w.mean()
                res_std_w = err_w.std(ddof=1)
                spear_w, _ = spearmanr(sub['pred_w'], sub[dft_w_col])

                # Tc stats
                err_tc = sub['tc_pred_guard'] - sub[dft_tc_col]
                mae_tc = np.abs(err_tc).mean()
                bias_tc = err_tc.mean()
                res_std_tc = err_tc.std(ddof=1)

                print(f"  [{regime}] (n = {n}):")
                print(f"    lambda: MAE = {mae_l:.4f}, bias = {bias_l:+.4f}, Spearman = {spear_l:.4f}, residual std = {res_std_l:.4f}")
                print(f"    w_log:  MAE = {mae_w:.2f} K, bias = {bias_w:+.2f} K, Spearman = {spear_w:.4f}, residual std = {res_std_w:.2f} K")
                print(f"    Tc:     MAE = {mae_tc:.2f} K, bias = {bias_tc:+.2f} K, residual std = {res_std_tc:.2f} K")

        return df_cal

    df_cal_test = run_calibration(matched_test_ids, "HELD-OUT TEST SET")
    df_cal_all = run_calibration(matched_all_ids, "FULL 806 DATASET")

    # -------------------------------------------------------------
    # STEP 4: BUCKETS (CONFUSION MATRIX)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. BUCKETS (Tc Confusion Matrix)")
    print("=" * 50)

    def assign_bucket(tc):
        if tc < 1.0:
            return "<1 K"
        elif tc <= 5.0:
            return "1-5 K"
        else:
            return ">5 K"

    bucket_labels = ["<1 K", "1-5 K", ">5 K"]

    def analyze_buckets(df_cal, name, dft_col='tc_dft_stored_guard'):
        df_cal['pred_bucket'] = df_cal['tc_pred_guard'].apply(assign_bucket)
        df_cal['dft_bucket'] = df_cal[dft_col].apply(assign_bucket)

        cm = pd.crosstab(df_cal['dft_bucket'], df_cal['pred_bucket'], rownames=['DFT Truth'], colnames=['Predicted'], dropna=False)
        for b in bucket_labels:
            if b not in cm.index:
                cm.loc[b] = 0
            if b not in cm.columns:
                cm[b] = 0
        cm = cm.reindex(index=bucket_labels, columns=bucket_labels, fill_value=0)

        print(f"\n--- CONFUSION MATRIX: {name} (Rows: DFT, Cols: Pred) ---")
        print(cm)

        precisions = {}
        print("\nPrecision per Predicted Bucket (Fraction where DFT matches Predicted):")
        for b in bucket_labels:
            total_pred = cm[b].sum()
            correct = cm.loc[b, b]
            prec = correct / total_pred if total_pred > 0 else 0.0
            precisions[b] = prec
            print(f"  Predicted {b:5s}: {correct:3d} / {total_pred:3d} ({prec*100:5.1f}%) match DFT")
        return precisions, cm

    prec_test_stored, cm_test_stored = analyze_buckets(df_cal_test, "HELD-OUT TEST SET (vs Stored DFT)", 'tc_dft_stored_guard')
    prec_test_recomp, cm_test_recomp = analyze_buckets(df_cal_test, "HELD-OUT TEST SET (vs Recomputed DFT)", 'tc_dft_recomp_guard')
    prec_all_stored, cm_all_stored = analyze_buckets(df_cal_all, "FULL 806 DATASET (vs Stored DFT)", 'tc_dft_stored_guard')

    # -------------------------------------------------------------
    # STEP 5: CANDIDATES
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("5. CANDIDATES EVALUATION (44 STABLE Candidates)")
    print("=" * 50)

    # Elemental exclusions for OOD: Co, Fe, Mn, Ni, Cu, Ru or rare earths
    rare_earths = {'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Sc', 'Y'}
    target_transition_metals = {'Co', 'Fe', 'Mn', 'Ni', 'Cu', 'Ru'}
    ood_elements = rare_earths | target_transition_metals

    # Map formulas from phonon_stability_results.csv or reference table
    cand_formulas = {}
    for idx, row in df_ph.iterrows():
        cand_formulas[row['material_id']] = row['formula']

    # Residual stds from Step 3 (Held-out Test Set, Stored DFT)
    # lambda < 0.4: std ~ 0.179; lambda >= 0.4: std ~ 0.536
    sub_low = df_cal_test[df_cal_test['dft_l_stored'] < 0.4]
    std_low = (sub_low['pred_l'] - sub_low['dft_l_stored']).std(ddof=1)
    sub_high = df_cal_test[df_cal_test['dft_l_stored'] >= 0.4]
    std_high = (sub_high['pred_l'] - sub_high['dft_l_stored']).std(ddof=1)
    std_overall = (df_cal_test['pred_l'] - df_cal_test['dft_l_stored']).std(ddof=1)

    print(f"Step 3 Residual Stds used for uncertainty:")
    print(f"  lambda < 0.4:  +- {std_low:.3f}")
    print(f"  lambda >= 0.4: +- {std_high:.3f}")
    print(f"  overall:       +- {std_overall:.3f}")

    cand_results = []
    for cid in stable_cand_ids:
        # Check if in mp_preds or ext_preds
        if cid in mp_preds:
            tc_s, tc_v, lam_v, wlog_v = mp_preds[cid]
        elif cid in {v[0] for v in ext_preds.values()}:
            # Find in ext_preds
            match = [v for v in ext_preds.values() if v[0] == cid][0]
            _, tc_v, lam_v, wlog_v = match
        else:
            print(f"Missing candidate: {cid}")
            continue

        form = cand_formulas.get(cid, "Unknown")

        # Check OOD elements
        # Extract elements using regex
        elements_in_form = re.findall(r'([A-Z][a-z]*)', form)
        present_ood = [el for el in elements_in_form if el in ood_elements]
        is_ood = len(present_ood) > 0

        # Guarded Tc
        tc_guarded = cal_tc_guarded(lam_v, wlog_v)
        b = assign_bucket(tc_guarded)
        prec = prec_test_stored.get(b, 0.0)

        res_std = std_high if lam_v >= 0.4 else std_low

        cand_results.append({
            'material_id': cid,
            'formula': form,
            'lambda': lam_v,
            'res_std': res_std,
            'w_log': wlog_v,
            'tc_guarded': tc_guarded,
            'bucket': b,
            'bucket_precision': prec,
            'is_ood': is_ood,
            'ood_reasons': ", ".join(present_ood) if is_ood else "IN-DOMAIN"
        })

    df_cands = pd.DataFrame(cand_results)
    
    # Sort by Tc descending
    df_cands = df_cands.sort_values(by='tc_guarded', ascending=False)

    print("\n" + "-" * 110)
    print(f"{'Material ID':14s} | {'Formula':12s} | {'lambda +- std':16s} | {'w_log (K)':9s} | {'Tc (K)':7s} | {'Bucket':6s} | {'Precision':9s} | {'Domain Status'}")
    print("-" * 110)
    for _, r in df_cands.iterrows():
        l_str = f"{r['lambda']:.3f} +- {r['res_std']:.3f}"
        prec_str = f"{r['bucket_precision']*100:.1f}%"
        status_str = f"OOD ({r['ood_reasons']})" if r['is_ood'] else "IN-DOMAIN"
        print(f"{r['material_id']:14s} | {r['formula']:12s} | {l_str:16s} | {r['w_log']:9.2f} | {r['tc_guarded']:7.3f} | {r['bucket']:6s} | {prec_str:9s} | {status_str}")
    print("-" * 110)

    print(f"\nTotal Candidates Evaluated: {len(df_cands)}")
    print(f"In-Domain Candidates: {len(df_cands[~df_cands['is_ood']])}")
    print(f"Out-of-Domain Candidates: {len(df_cands[df_cands['is_ood']])}")
    print(f"Candidates with Tc >= 5 K: {len(df_cands[df_cands['tc_guarded'] >= 5.0])}")
    print(f"In-Domain Candidates with Tc >= 5 K: {len(df_cands[(df_cands['tc_guarded'] >= 5.0) & (~df_cands['is_ood'])])}")

if __name__ == '__main__':
    main()
