import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import ase.io
import torch_geometric as tg
from notebooks.utils.data import build_data, get_neighbors, get_target
from notebooks.utils.training import get_model

def parse_id(raw_id):
    """Safely parse IDs like '7.100000000000000000e+01' to integer ID string '71'."""
    try:
        return str(int(float(raw_id.strip())))
    except Exception:
        return raw_id.strip()

Freq_final = np.arange(0.25, 101, 2)

def cal_lamb(freq_w, alpha_F):
    lambdaF = 0.0
    for i in range(1, len(freq_w)):
        dw = freq_w[i] - freq_w[i-1]
        w = freq_w[i]
        lambdaF += (alpha_F[i] / w) * dw
    return 2.0 * lambdaF

def cal_w_log(freq_w, alpha_F, lambdaF):
    if lambdaF <= 0.0:
        return 0.0
    w_log = 0.0
    for i in range(1, len(freq_w)):
        dw = freq_w[i] - freq_w[i-1]
        w = freq_w[i]
        w_log += (alpha_F[i] / w) * np.log(w) * dw
    w_log = (2.0 / lambdaF) * w_log
    # convert meV to Kelvin: / 0.08617 meV/K
    return float(np.exp(w_log) / 0.08617)

def cal_w_sq(freq_w, alpha_F, lambdaF):
    if lambdaF <= 0.0:
        return 0.0
    w_sq = 0.0
    for i in range(1, len(freq_w)):
        dw = freq_w[i] - freq_w[i-1]
        w = freq_w[i]
        w_sq += (alpha_F[i] * w) * dw
    w_sq = (2.0 / lambdaF) * w_sq
    return float(np.sqrt(w_sq) / 0.08617)

def cal_tc(lamb, wlog, mu=0.1):
    denom = lamb - mu * (1.0 + 0.62 * lamb)
    if denom <= 0.0 or lamb <= 0.0 or wlog <= 0.0:
        return 0.0
    tc = (wlog / 1.2) * np.exp(-1.04 * (1.0 + lamb) / denom)
    return float(tc)

def cal_tc_ad(lamb, wlog, w2, tc_base, mu=0.1):
    if tc_base <= 0.0 or lamb <= 0.0 or wlog <= 0.0 or w2 <= 0.0:
        return 0.0
    denom = lamb - mu * (1.0 + 0.62 * lamb)
    if denom <= 0.0:
        return 0.0
    f1 = (1.0 + (lamb / (2.46 * (1.0 + 3.8 * mu))) ** 1.5) ** (1.0 / 3.0)
    w2_wlog = w2 / wlog
    f2 = 1.0 + ((w2_wlog - 1.0) * lamb**2) / (lamb**2 + (1.82 * (1.0 + 6.3 * mu) * (w2_wlog)) ** 2)
    return float(f1 * f2 * tc_base)

def main():
    start_time = time.time()
    print("=== BETE-NET FULL BATCH RUN (CSO ENSEMBLE, 100 CHECKPOINTS) ===")
    
    # 1. Read original database.json and compute live train_num_neighbors BEFORE anything can modify it
    print("\n--- Step 1: Live train_num_neighbors from pristine database.json ---")
    assert os.path.exists('database.json'), "Original database.json not found! Run git checkout -- database.json first."
    orig_df = pd.read_json('database.json')
    assert len(orig_df) == 806, f"Expected 806 rows in database.json, got {len(orig_df)}"
    
    orig_structures = []
    for index, row in orig_df.iterrows():
        orig_structures.append(ase.io.read(f'structures/{index}.cif'))
    orig_df['structure'] = orig_structures
    orig_df['target'] = orig_df.apply(get_target, axis=1)
    orig_df['data'] = orig_df.apply(build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4, axis=1)
    
    train_num_neighbors = float(get_neighbors(orig_df, orig_df.index).mean())
    print(f"Live train_num_neighbors computed: {train_num_neighbors:.4f}")
    assert abs(train_num_neighbors - 16.8463) < 0.01, f"Expected within 0.01 of 16.8463, got {train_num_neighbors:.4f}"
    
    # 2. Setup run structure IDs with parse_id()
    print("\n--- Step 2: Assembling run structure IDs ---")
    idx_test_path = "indices/idx_test_full.txt"
    assert os.path.exists(idx_test_path), f"Error: {idx_test_path} not found."
    with open(idx_test_path, 'r') as f:
        raw_test_lines = [line.strip() for line in f if line.strip()]
        
    parsed_test_ids = [parse_id(line) for line in raw_test_lines]
    
    # Check dropped test IDs
    dropped_test_ids = [tid for tid in parsed_test_ids if not os.path.exists(f'structures/{tid}.cif')]
    held_out_ids = [tid for tid in parsed_test_ids if os.path.exists(f'structures/{tid}.cif')]
    
    print(f"Total lines in idx_test_full.txt: {len(parsed_test_ids)}")
    print(f"Dropped test IDs missing CIF: {dropped_test_ids}")
    assert set(dropped_test_ids) == {'36', '786', '218'}, f"Unexpected dropped test IDs: {dropped_test_ids}"
    assert len(held_out_ids) == 170, f"Expected 170 held-out IDs with CIFs, got {len(held_out_ids)}"
    
    ext_refs = {
        "mp-aaacrmzn": "Mo2CN",
        "mp-aaacrral": "HfZrN2",
        "mp-aaaaaghy": "BeAlB",
        "mp-aaacpwsj_niggli": "Nb8PtSe20 (Niggli)",
        "mp-aaacpwsj": "Nb8PtSe20 (Skewed)"
    }
    
    all_cifs = [f for f in os.listdir('structures') if f.endswith('.cif')]
    
    # Order: held-out test IDs first, then external references, then remaining candidate structures
    run_ids = held_out_ids + list(ext_refs.keys()) + [
        f.replace('.cif', '') for f in all_cifs
        if f.replace('.cif', '') not in held_out_ids and f.replace('.cif', '') not in ext_refs.keys()
    ]
    # Remove duplicates while preserving order
    run_ids = list(dict.fromkeys(run_ids))
    
    n = len(run_ids)
    print(f"Total structures to process: n = {n}")
    print(f"First 5 IDs to process: {run_ids[:5]}")
    assert n <= 900, f"Total structures n={n} exceeds single batch size limit 900!"
    
    # 3. Build evaluation DataFrame without dropna
    records = []
    for sid in run_ids:
        cif_path = f'structures/{sid}.cif'
        assert os.path.exists(cif_path), f"Structure CIF {cif_path} missing!"
        st = ase.io.read(cif_path)
        
        # Determine human-readable formula
        if sid in ext_refs:
            formula = ext_refs[sid]
        elif sid.isdigit() and int(sid) in orig_df.index:
            formula = orig_df.loc[int(sid), 'comp']
        else:
            formula = st.get_chemical_formula()
            
        records.append({
            'index': sid,
            'structure': st,
            'formula': formula,
            'target': np.zeros(51, dtype=np.float32).tolist()
        })
        
    df = pd.DataFrame(records)
    df.set_index('index', inplace=True)
    
    assert len(df) == len(run_ids), f"len(df) ({len(df)}) != len(run_ids) ({len(run_ids)})! Dropping occurred."
    print(f"DataFrame built successfully: len(df) = {len(df)} == {len(run_ids)}.")
    
    # 4. Build graph data
    print("\n--- Step 3: Building data graphs ---")
    print(f"Using verified train_num_neighbors = {train_num_neighbors:.4f}")
    df['data'] = df.apply(build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4, axis=1)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    
    out_dim = 51
    in_dim = len(df.iloc[0].data.x[0])
    em_dim = 64
    
    init_dict_base = dict(
        in_dim=118,
        em_dim=em_dim,
        irreps_in=str(em_dim) + "x0e",
        irreps_out=str(out_dim) + "x0e",
        irreps_node_attr=str(em_dim) + "x0e",
        layers=2,
        mul=32,
        lmax=1,
        max_radius=4,
        num_neighbors=train_num_neighbors,
        reduce_output=True,
        p=0.0
    )
    
    dataloader = tg.loader.DataLoader(df['data'].values, batch_size=900)
    
    # 5. Checkpoint evaluation loop over 100 checkpoints
    folds = range(100)
    ensemble_preds = np.zeros((100, n, out_dim), dtype=np.float64)
    
    print("\n--- Step 4: Running inference across 100 CSO checkpoints ---")
    loop_start = time.time()
    for k in folds:
        name = f"model_cso_{k}.pt"
        run_name = f'CSO/{name}'
        assert os.path.exists(run_name), f"Checkpoint {run_name} not found!"
        
        model, opt, scheduler = get_model(init_dict_base, device=device)
        model.load_state_dict(torch.load(run_name, map_location=device))
        model.pool = True
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for d in dataloader:
                d.to(device)
                output = model(d)
                ensemble_preds[k] = output.cpu().numpy()
                df[f'pred_{k}'] = [val for val in ensemble_preds[k]]
                
        elapsed = time.time() - loop_start
        eta = (elapsed / (k + 1)) * (100 - (k + 1))
        print(f"[{k+1:3d}/100] Checkpoint {name} completed. Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s", flush=True)
        
    # 6. Save ensemble predictions array named from real n
    npy_filename = f"cso_ensemble_preds_{n}.npy"
    np.save(npy_filename, ensemble_preds)
    print(f"\nSaved full ensemble predictions array to {npy_filename} (shape {ensemble_preds.shape}).")
    
    # 7. Compute spectral parameters and guarded Tc
    print("\n--- Step 5: Computing lambda, w_log, and guarded Tc ---")
    pred_avg = np.mean(ensemble_preds, axis=0)
    df['pred_avg'] = [p for p in pred_avg]
    
    lamb_list = []
    wlog_list = []
    w2_list = []
    tc_list = []
    tcad_list = []
    tc_std_list = []
    guarded_count = 0
    
    for i in range(n):
        l = cal_lamb(Freq_final, pred_avg[i])
        wl = cal_w_log(Freq_final, pred_avg[i], l)
        w2 = cal_w_sq(Freq_final, pred_avg[i], l)
        
        # Guard condition: return 0.0 if lamb <= mu*(1 + 0.62*lamb) or non-positive
        mu_val = 0.1
        denom = l - mu_val * (1.0 + 0.62 * l)
        if denom <= 0.0 or l <= 0.0 or wl <= 0.0:
            tc = 0.0
            tcad = 0.0
            guarded_count += 1
        else:
            tc = cal_tc(l, wl, mu=mu_val)
            tcad = cal_tc_ad(l, wl, w2, tc, mu=mu_val)
            
        # Ensemble std of Tc
        member_tcs = []
        for k in folds:
            m_spec = ensemble_preds[k, i]
            m_l = cal_lamb(Freq_final, m_spec)
            m_wl = cal_w_log(Freq_final, m_spec, m_l)
            m_denom = m_l - mu_val * (1.0 + 0.62 * m_l)
            if m_denom <= 0.0 or m_l <= 0.0 or m_wl <= 0.0:
                member_tcs.append(0.0)
            else:
                member_tcs.append(cal_tc(m_l, m_wl, mu=mu_val))
        tc_std = float(np.std(member_tcs))
        
        lamb_list.append(l)
        wlog_list.append(wl)
        w2_list.append(w2)
        tc_list.append(tc)
        tcad_list.append(tcad)
        tc_std_list.append(tc_std)
        
    df['lambda'] = lamb_list
    df['w_log'] = wlog_list
    df['w_2'] = w2_list
    df['Tc'] = tc_list
    df['Tc_AD'] = tcad_list
    df['Tc_std'] = tc_std_list
    
    print(f"Tc Guard Report: {guarded_count}/{n} structures guarded to Tc = 0.0 K.")
    assert np.all(np.isfinite(df['Tc'])), "Non-finite Tc detected!"
    assert np.all(df['Tc'] < 100.0), f"Tc >= 100 K detected! Max Tc = {df['Tc'].max()}"
    print(f"Tc assertion passed: all values finite and < 100 K (Max Tc = {df['Tc'].max():.2f} K).")
    
    # 8. Comparison with published CSO.json for the 170 held-out IDs
    print("\n--- Step 6: Comparison with Published CSO.json (170 Held-out IDs) ---")
    assert os.path.exists('test_preds/CSO.json'), "test_preds/CSO.json not found!"
    with open('test_preds/CSO.json', 'r') as f:
        cso_pub = json.load(f)
        
    within_lamb = 0
    within_wlog = 0
    within_both = 0
    lamb_diffs = []
    wlog_diffs = []
    
    for tid in held_out_ids:
        our_l = df.loc[tid, 'lambda']
        our_wl = df.loc[tid, 'w_log']
        
        pub_l = float(cso_pub['lamb_pred'][str(tid)])
        pub_wl = float(cso_pub['wlog_pred'][str(tid)])
        
        diff_l = abs(our_l - pub_l)
        diff_wl = abs(our_wl - pub_wl)
        lamb_diffs.append(diff_l)
        wlog_diffs.append(diff_wl)
        
        is_l = diff_l <= 2e-3
        is_wl = diff_wl <= 0.5
        if is_l:
            within_lamb += 1
        if is_wl:
            within_wlog += 1
        if is_l and is_wl:
            within_both += 1
            
    frac_l = (within_lamb / 170.0) * 100.0
    frac_wl = (within_wlog / 170.0) * 100.0
    frac_both = (within_both / 170.0) * 100.0
    
    print(f"Held-out IDs (n=170) within 2e-3 in lambda: {within_lamb}/170 ({frac_l:.1f}%)")
    print(f"Held-out IDs (n=170) within 0.5 K in w_log: {within_wlog}/170 ({frac_wl:.1f}%)")
    print(f"Held-out IDs (n=170) within BOTH thresholds: {within_both}/170 ({frac_both:.1f}%)")
    print(f"Mean abs diff: lambda = {np.mean(lamb_diffs):.4f}, w_log = {np.mean(wlog_diffs):.2f} K")
    print(f"Max abs diff:  lambda = {np.max(lamb_diffs):.4f}, w_log = {np.max(wlog_diffs):.2f} K")
    
    # 9. Save bete_net_results.csv
    csv_cols = ['formula', 'lambda', 'w_log', 'w_2', 'Tc', 'Tc_AD', 'Tc_std']
    df_out = df[csv_cols].copy()
    df_out.index.name = 'id'
    out_csv = 'bete_net_results.csv'
    df_out.to_csv(out_csv)
    print(f"\nSaved results table to {out_csv} ({len(df_out)} rows).")
    
    # Print external reference results
    print("\n--- External Reference Predictions ---")
    for ref_id, ref_name in ext_refs.items():
        if ref_id in df.index:
            r = df.loc[ref_id]
            print(f"{ref_name:<20} ({ref_id}): lambda={r['lambda']:.4f}, w_log={r['w_log']:.2f} K, Tc={r['Tc']:.2f} K +/- {r['Tc_std']:.2f} K")
            
    total_time = time.time() - start_time
    print(f"\nBatch run finished in {total_time:.2f}s.")

if __name__ == '__main__':
    main()
