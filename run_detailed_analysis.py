import json
import re
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, average_precision_score

sys.stdout.reconfigure(encoding='utf-8')

# Wilson score interval for binomial proportion
def wilson_interval(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt((p * (1.0 - p) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))

def cal_tc_guarded(lamb, omega_log, mu=0.09):
    denom = lamb - mu * (1.0 + 0.62 * lamb)
    if denom <= 0 or np.isnan(lamb) or np.isnan(omega_log) or lamb <= 0 or omega_log <= 0:
        return 0.0
    frac = -1.04 * (1.0 + lamb) / denom
    tc = (omega_log / 1.2) * np.exp(frac)
    return tc if np.isfinite(tc) else 0.0

def assign_bucket(tc):
    if tc < 1.0:
        return "<1 K"
    elif tc <= 5.0:
        return "1-5 K"
    else:
        return ">5 K"

def main():
    print("================================================================================")
    print("DETAILED BETE-NET AUDIT: DIFF, OFFSET, CALIBRATION, FUNNEL, AND DOMAIN")
    print("================================================================================")

    # Load 170 held-out IDs
    with open('BETE-NET-zip/BETE-NET-main/indices/idx_test_full.txt', 'r') as f:
        idx_test_raw = [line.strip() for line in f if line.strip()]
    idx_test_ids = [str(int(float(x))) for x in idx_test_raw]

    # Load database.json
    with open('BETE-NET-zip/BETE-NET-main/database.json', 'r', encoding='utf-8') as f:
        db = json.load(f)

    # Load CSO.json
    with open('BETE-NET-zip/BETE-NET-main/test_preds/CSO.json', 'r', encoding='utf-8') as f:
        cso = json.load(f)

    # Parse colab_run_856.txt
    colab_preds = {}
    mp_preds = {}
    ext_preds = {}
    with open('colab_run_856.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            m_ext = re.match(r'^(.*?)\s*\((mp-[a-z0-9_]+)\):\s*Tc=([^,]+),\s*lambda=([^,]+),\s*w_log=([^\s]+)\s*K', line)
            if m_ext:
                name, mpid, tc_s, lam_s, wlog_s = m_ext.groups()
                ext_preds[name] = (mpid, float(tc_s.replace('K','').strip()), float(lam_s), float(wlog_s))
                continue
            m = re.match(r'^([^:]+):\s*Tc=([^,]+),\s*lambda=([^,]+),\s*w_log=([^\s]+)\s*K', line)
            if m:
                id_s, tc_s, lam_s, wlog_s = m.groups()
                id_s = id_s.strip()
                tc_c = tc_s.replace('K','').strip()
                tc_v = float(tc_c) if tc_c != 'inf' else np.inf
                lam_v = float(lam_s.strip())
                wlog_v = float(wlog_s.strip())
                if id_s.startswith('mp-'):
                    mp_preds[id_s] = (tc_v, lam_v, wlog_v)
                elif id_s.isdigit():
                    colab_preds[id_s] = (tc_v, lam_v, wlog_v)

    matched_170 = [tid for tid in idx_test_ids if tid in colab_preds and tid in cso['lamb_pred']]
    print(f"Matched held-out test IDs: {len(matched_170)} / 173")

    # Build DataFrame for 170 held-out IDs
    records = []
    for tid in matched_170:
        c_tc, c_l, c_w = colab_preds[tid]
        p_l = cso['lamb_pred'][tid]
        p_w = cso['wlog_pred'][tid]
        p_tc = cal_tc_guarded(p_l, p_w)
        c_tc_g = cal_tc_guarded(c_l, c_w)

        dft_l = db['lambda'][tid]
        dft_w = db['w_log'][tid] / 0.08617
        dft_tc = cal_tc_guarded(dft_l, dft_w)

        records.append({
            'id': tid,
            'comp': db['comp'][tid],
            'ours_l': c_l,
            'ours_w': c_w,
            'ours_tc': c_tc_g,
            'cso_l': p_l,
            'cso_w': p_w,
            'cso_tc': p_tc,
            'dft_l': dft_l,
            'dft_w': dft_w,
            'dft_tc': dft_tc
        })
    df_170 = pd.DataFrame(records)

    # -------------------------------------------------------------
    # ITEM 2: OFFSET ANALYSIS (Ours vs CSO.json on 170 IDs)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ITEM 2: OFFSET ANALYSIS (Ours vs CSO.json on 170 held-out IDs)")
    print("=" * 60)

    # Lambda offset
    diff_l = df_170['ours_l'] - df_170['cso_l']
    mean_diff_l = diff_l.mean()
    median_diff_l = diff_l.median()
    reg_l = linregress(df_170['cso_l'], df_170['ours_l'])
    pred_reg_l = reg_l.intercept + reg_l.slope * df_170['cso_l']
    res_std_l = (df_170['ours_l'] - pred_reg_l).std(ddof=2)

    # w_log offset
    diff_w = df_170['ours_w'] - df_170['cso_w']
    mean_diff_w = diff_w.mean()
    median_diff_w = diff_w.median()
    reg_w = linregress(df_170['cso_w'], df_170['ours_w'])
    pred_reg_w = reg_w.intercept + reg_w.slope * df_170['cso_w']
    res_std_w = (df_170['ours_w'] - pred_reg_w).std(ddof=2)

    print("Lambda (ours vs CSO.json):")
    print(f"  Signed Mean Error (ours - CSO):   {mean_diff_l:+.6f}")
    print(f"  Signed Median Error (ours - CSO): {median_diff_l:+.6f}")
    print(f"  Regression: ours = {reg_l.slope:.6f} * CSO + ({reg_l.intercept:+.6f}) (R^2 = {reg_l.rvalue**2:.6f})")
    print(f"  Residual Std after slope:         {res_std_l:.6f}")

    print("\nw_log (ours vs CSO.json):")
    print(f"  Signed Mean Error (ours - CSO):   {mean_diff_w:+.6f} K")
    print(f"  Signed Median Error (ours - CSO): {median_diff_w:+.6f} K")
    print(f"  Regression: ours = {reg_w.slope:.6f} * CSO + ({reg_w.intercept:+.6f}) (R^2 = {reg_w.rvalue**2:.6f})")
    print(f"  Residual Std after slope:         {res_std_w:.6f} K")

    # -------------------------------------------------------------
    # ITEM 3: TWICE (Held-out Calibration: Ours vs CSO.json)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ITEM 3: TWICE - CALIBRATION SIDE-BY-SIDE (Ours vs CSO.json)")
    print("=" * 60)

    for name, p_l_col, p_w_col, p_tc_col in [("OURS (colab_run_856.txt)", "ours_l", "ours_w", "ours_tc"),
                                            ("CSO.json (Published)", "cso_l", "cso_w", "cso_tc")]:
        err_l = df_170[p_l_col] - df_170['dft_l']
        err_w = df_170[p_w_col] - df_170['dft_w']
        err_tc = df_170[p_tc_col] - df_170['dft_tc']

        sp_l, _ = spearmanr(df_170[p_l_col], df_170['dft_l'])
        sp_w, _ = spearmanr(df_170[p_w_col], df_170['dft_w'])

        print(f"\n--- {name} (n = {len(df_170)}) ---")
        print(f"  Lambda: MAE = {err_l.abs().mean():.4f}, Bias = {err_l.mean():+.4f}, Spearman = {sp_l:.4f}, Res Std = {err_l.std(ddof=1):.4f}")
        print(f"  w_log:  MAE = {err_w.abs().mean():.2f} K, Bias = {err_w.mean():+.2f} K, Spearman = {sp_w:.4f}, Res Std = {err_w.std(ddof=1):.2f} K")
        print(f"  Tc:     MAE = {err_tc.abs().mean():.2f} K, Bias = {err_tc.mean():+.2f} K, Res Std = {err_tc.std(ddof=1):.2f} K")

        df_170['p_b'] = df_170[p_tc_col].apply(assign_bucket)
        df_170['d_b'] = df_170['dft_tc'].apply(assign_bucket)
        cm = pd.crosstab(df_170['d_b'], df_170['p_b'], dropna=False)
        for b in ["<1 K", "1-5 K", ">5 K"]:
            if b not in cm.index: cm.loc[b] = 0
            if b not in cm.columns: cm[b] = 0
        cm = cm.reindex(index=["<1 K", "1-5 K", ">5 K"], columns=["<1 K", "1-5 K", ">5 K"], fill_value=0)
        print("  Tc Confusion Matrix (Rows: DFT, Cols: Pred):")
        print(cm)
        print("  Precision per Predicted Bucket:")
        for b in ["<1 K", "1-5 K", ">5 K"]:
            tot = cm[b].sum()
            cor = cm.loc[b, b]
            print(f"    Predicted {b:5s}: {cor:2d}/{tot:3d} ({cor/tot*100:5.1f}%)" if tot > 0 else f"    Predicted {b}: 0")

    # -------------------------------------------------------------
    # ITEM 4: CONDITION ON PREDICTION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ITEM 4: CONDITION ON PREDICTION (Held-Out Test Set)")
    print("=" * 60)

    # Binning by PREDICTED lambda: (<0.3, 0.3-0.4, 0.4-0.5, 0.5-0.7, >0.7)
    bins = [0.0, 0.3, 0.4, 0.5, 0.7, 10.0]
    bin_labels = ["<0.3", "0.3-0.4", "0.4-0.5", "0.5-0.7", ">0.7"]
    df_170['l_bin'] = pd.cut(df_170['ours_l'], bins=bins, labels=bin_labels, right=False)

    bin_stats = {}
    print(f"{'Pred Bin':10s} | {'n':4s} | {'Mean Pred':9s} | {'Mean DFT':9s} | {'Bias (P-D)':10s} | {'Res Std':8s}")
    print("-" * 65)
    for b in bin_labels:
        sub = df_170[df_170['l_bin'] == b]
        n_b = len(sub)
        mean_p = sub['ours_l'].mean() if n_b > 0 else 0
        mean_d = sub['dft_l'].mean() if n_b > 0 else 0
        bias = (sub['ours_l'] - sub['dft_l']).mean() if n_b > 0 else 0
        res_std = (sub['dft_l'] - sub['ours_l']).std(ddof=1) if n_b > 1 else 0
        bin_stats[b] = {'n': n_b, 'mean_p': mean_p, 'mean_d': mean_d, 'bias': bias, 'std': res_std}
        print(f"{b:10s} | {n_b:4d} | {mean_p:9.4f} | {mean_d:9.4f} | {bias:+10.4f} | {res_std:8.4f}")

    # P(DFT bucket | Predicted bucket) with Wilson 95% intervals
    df_170['p_b'] = df_170['ours_tc'].apply(assign_bucket)
    df_170['d_b'] = df_170['dft_tc'].apply(assign_bucket)

    print("\nP(DFT bucket | Predicted bucket) with Wilson 95% Confidence Intervals:")
    print(f"{'Pred Bucket':12s} | {'DFT Bucket':10s} | {'Count / Total':14s} | {'P(DFT|Pred)':12s} | {'Wilson 95% CI'}")
    print("-" * 70)
    for pb in ["<1 K", "1-5 K", ">5 K"]:
        sub_p = df_170[df_170['p_b'] == pb]
        n_tot = len(sub_p)
        for db_cat in ["<1 K", "1-5 K", ">5 K"]:
            k = (sub_p['d_b'] == db_cat).sum()
            p = k / n_tot if n_tot > 0 else 0
            ci_low, ci_high = wilson_interval(k, n_tot)
            print(f"Pred {pb:6s} | DFT {db_cat:6s} | {k:3d} / {n_tot:3d}     | {p*100:6.1f}%      | [{ci_low*100:5.1f}%, {ci_high*100:5.1f}%]")

    # -------------------------------------------------------------
    # ITEM 5: FUNNEL (ROC AUC, PR AUC, TOP-K ENRICHMENT)
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ITEM 5: FUNNEL ANALYSIS (Held-Out Test Set n = 170)")
    print("=" * 60)

    n_total = len(df_170)
    cutoffs = [("Top 5%", int(np.ceil(0.05 * n_total))),
               ("Top 10%", int(np.ceil(0.10 * n_total))),
               ("Top 25%", int(np.ceil(0.25 * n_total)))]

    for threshold, label_col in [(5.0, "DFT Tc > 5 K"), (1.0, "DFT Tc > 1 K")]:
        y_true = (df_170['dft_tc'] > threshold).astype(int).values
        base_rate = y_true.mean()
        total_positives = y_true.sum()

        print(f"\n================ Target: {label_col} (Positives: {total_positives} / {n_total}, Base Rate: {base_rate*100:.1f}%) ================")

        for score_name, score_col in [("Tc_pred", "ours_tc"), ("lambda_pred", "ours_l"), ("w_log_pred", "ours_w")]:
            scores = df_170[score_col].values
            roc_auc = roc_auc_score(y_true, scores)
            pr_auc = average_precision_score(y_true, scores)

            # Sort descending
            order = np.argsort(-scores)
            y_sorted = y_true[order]

            print(f"\nRanked by {score_name}:")
            print(f"  ROC AUC: {roc_auc:.4f} | PR AUC (Avg Precision): {pr_auc:.4f}")
            print(f"  Cutoff   | Count | Hits | Precision | Recall | Enrichment")
            print(f"  " + "-" * 55)

            for cut_name, k in cutoffs:
                hits = y_sorted[:k].sum()
                prec = hits / k
                rec = hits / total_positives if total_positives > 0 else 0
                enrich = prec / base_rate if base_rate > 0 else 0
                print(f"  {cut_name:8s} | {k:5d} | {hits:4d} | {prec*100:8.1f}% | {rec*100:5.1f}% | {enrich:6.2f}x")

    # -------------------------------------------------------------
    # ITEM 6: DOMAIN & CANDIDATES RE-EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ITEM 6: DOMAIN & CANDIDATE POOL (Prevalence in Database & Flips)")
    print("=" * 60)

    # Count elements in database.json (all 806) and 170 held-out
    # Collect formulas from db
    all_elements_db = {}
    test_elements_db = {}
    for tid, c in db['comp'].items():
        els = re.findall(r'([A-Z][a-z]*)', c)
        for el in set(els):
            all_elements_db[el] = all_elements_db.get(el, 0) + 1
            if tid in matched_170:
                test_elements_db[el] = test_elements_db.get(el, 0) + 1

    # Load 44 candidate formulas
    df_ph = pd.read_csv('phonon_stability_results.csv')
    stable_cand = df_ph[df_ph['status'] == 'STABLE']

    # Get elements across all 44 candidates
    cand_elements = sorted(list(set(re.findall(r'([A-Z][a-z]*)', "".join(stable_cand['formula'])))))

    print("Element prevalence in BETE-NET database:")
    print(f"{'Element':8s} | {'Count in 806':12s} | {'Count in 170':12s} | {'Status (<10 = Rare/OOD)'}")
    print("-" * 55)
    for el in cand_elements:
        c_all = all_elements_db.get(el, 0)
        c_test = test_elements_db.get(el, 0)
        magnetic = "MAGNETIC (OOD)" if el in ['Fe', 'Co', 'Mn', 'Ni'] else ""
        rare = "RARE (<10 in db)" if c_all < 10 else "PRESENT (>=10 in db)"
        flag = magnetic if magnetic else rare
        print(f"{el:8s} | {c_all:12d} | {c_test:12d} | {flag}")

    # Re-evaluate candidate domain status:
    # Rule: Out of Domain if contains ANY element with count in 806 < 10 OR element in ['Fe', 'Co', 'Mn', 'Ni']
    print("\nRe-evaluating 44 candidates with rule (count in db < 10 OR Fe/Co/Mn/Ni):")
    cand_eval = []
    for idx, row in stable_cand.iterrows():
        cid = row['material_id']
        form = row['formula']
        els = set(re.findall(r'([A-Z][a-z]*)', form))

        # Check values
        if cid in mp_preds:
            tc_v, lam_v, wlog_v = mp_preds[cid]
        elif cid in {v[0] for v in ext_preds.values()}:
            match = [v for v in ext_preds.values() if v[0] == cid][0]
            _, tc_v, lam_v, wlog_v = match
        else:
            continue

        # Bin and uncertainty from ITEM 4
        # bins: [<0.3, 0.3-0.4, 0.4-0.5, 0.5-0.7, >0.7]
        if lam_v < 0.3:
            b_name = "<0.3"
        elif lam_v < 0.4:
            b_name = "0.3-0.4"
        elif lam_v < 0.5:
            b_name = "0.4-0.5"
        elif lam_v < 0.7:
            b_name = "0.5-0.7"
        else:
            b_name = ">0.7"
        std_bin = bin_stats[b_name]['std']

        tc_g = cal_tc_guarded(lam_v, wlog_v)
        b_tc = assign_bucket(tc_g)

        # Domain checks
        magnetic_els = [e for e in els if e in ['Fe', 'Co', 'Mn', 'Ni']]
        rare_els = [e for e in els if all_elements_db.get(e, 0) < 10]

        is_ood_new = len(magnetic_els) > 0 or len(rare_els) > 0
        reasons = []
        if magnetic_els: reasons.append(f"Magnetic: {','.join(magnetic_els)}")
        if rare_els: reasons.append(f"<10 in db: {','.join([f'{e}({all_elements_db.get(e,0)})' for e in rare_els])}")

        cand_eval.append({
            'id': cid,
            'formula': form,
            'lambda': lam_v,
            'bin': b_name,
            'std_bin': std_bin,
            'w_log': wlog_v,
            'tc_g': tc_g,
            'b_tc': b_tc,
            'is_ood': is_ood_new,
            'reason': "; ".join(reasons) if is_ood_new else "IN-DOMAIN"
        })

    df_cand_eval = pd.DataFrame(cand_eval).sort_values(by='tc_g', ascending=False)

    print(f"\nTotal Candidates: {len(df_cand_eval)}")
    print(f"New In-Domain Candidates: {len(df_cand_eval[~df_cand_eval['is_ood']])}")
    print(f"New Out-of-Domain Candidates: {len(df_cand_eval[df_cand_eval['is_ood']])}")
    print(f"Candidates with Tc >= 5 K: {len(df_cand_eval[df_cand_eval['tc_g'] >= 5.0])}")
    print(f"In-Domain Candidates with Tc >= 5 K: {len(df_cand_eval[(df_cand_eval['tc_g'] >= 5.0) & (~df_cand_eval['is_ood'])])}")

    print("\nCandidates Table (sorted by Tc descending):")
    print(f"{'Material ID':14s} | {'Formula':12s} | {'lambda +- std':16s} | {'w_log (K)':9s} | {'Tc (K)':7s} | {'Bucket':6s} | {'Domain Status'}")
    print("-" * 105)
    for _, r in df_cand_eval.iterrows():
        l_str = f"{r['lambda']:.3f} +- {r['std_bin']:.3f}"
        print(f"{r['id']:14s} | {r['formula']:12s} | {l_str:16s} | {r['w_log']:9.2f} | {r['tc_g']:7.3f} | {r['b_tc']:6s} | {r['reason']}")

if __name__ == '__main__':
    main()
