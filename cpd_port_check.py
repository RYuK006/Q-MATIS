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

def main():
    start_time = time.time()
    print("=== CPD PORT CHECK (20 HELD-OUT IDS, DFPT DOS, 100 CHECKPOINTS) ===")
    
    # 1. Verify environment and pristine database.json
    assert os.path.exists('database.json'), "database.json not found! Run git checkout -- database.json first."
    assert os.path.exists('indices/idx_test_full.txt'), "indices/idx_test_full.txt not found!"
    assert os.path.exists('test_preds/CPD.json'), "test_preds/CPD.json not found!"
    assert os.path.isdir('CPD'), "CPD/ directory not found!"
    
    # 2. Read original database.json and compute live train_num_neighbors
    print("\n--- Step 1: Live train_num_neighbors from pristine database.json ---")
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
    
    # 3. Read test indices and isolate 20 held-out IDs
    print("\n--- Step 2: Selecting 20 held-out test IDs ---")
    with open('indices/idx_test_full.txt', 'r') as f:
        raw_test_lines = [line.strip() for line in f if line.strip()]
        
    parsed_test_ids = [parse_id(line) for line in raw_test_lines]
    print(f"Total lines in idx_test_full.txt: {len(parsed_test_ids)}")
    
    # Pruned IDs check: exactly {36, 786, 218} should be missing from database.json
    dropped_test_ids = [tid for tid in parsed_test_ids if int(tid) not in orig_df.index]
    print(f"Pruned test IDs not in database.json: {dropped_test_ids}")
    assert set(dropped_test_ids) == {'36', '786', '218'}, f"Unexpected dropped test IDs: {dropped_test_ids}"
    
    held_out_ids = [tid for tid in parsed_test_ids if int(tid) in orig_df.index]
    assert len(held_out_ids) == 170, f"Expected 170 held-out IDs, got {len(held_out_ids)}"
    
    test_20 = held_out_ids[:20]
    print(f"Selected 20 held-out test IDs: {test_20}")
    assert len(test_20) == 20, f"Expected exactly 20 test IDs, got {len(test_20)}"
    
    # 4. Stream / load CPD.json and assert composition matching
    print("\n--- Step 3: Loading CPD.json and verifying compositions ---")
    with open('test_preds/CPD.json', 'r') as f:
        cpd_pub = json.load(f)
        
    for tid in test_20:
        db_comp = orig_df.loc[int(tid), 'comp']
        assert str(tid) in cpd_pub['comp'], f"ID {tid} missing from CPD.json 'comp'"
        cpd_comp = cpd_pub['comp'][str(tid)]
        assert db_comp == cpd_comp, f"Composition mismatch for ID {tid}: DB '{db_comp}' vs CPD '{cpd_comp}'"
    print("All 20 test compositions match CPD.json perfectly.")
    
    # 5. Build CPD evaluation dataframe with DFPT phonon DOS
    print("\n--- Step 4: Building graph data with DFPT coarse phonon DOS ---")
    records = []
    for tid in test_20:
        row = orig_df.loc[int(tid)]
        records.append({
            'index': int(tid),
            'structure': row['structure'],
            'comp': row['comp'],
            'target': row['target'],
            'Ph_2x2x2_interpolated_Freq_meV': row['Ph_2x2x2_interpolated_Freq_meV'],
            'Ph_2x2x2_interpolated_Site_Proj_DOS': row['Ph_2x2x2_interpolated_Site_Proj_DOS']
        })
    df_eval = pd.DataFrame(records)
    df_eval.set_index('index', inplace=True)
    
    # Build graph data with embed_ph_dos=True
    df_eval['data'] = df_eval.apply(build_data, embed_ph_dos=True, embed_e_dos=False, fine=False, r_max=4, axis=1)
    
    # 6. Setup CPD model configuration and DataLoader
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")
    
    out_dim = 51
    in_dim = len(df_eval.iloc[0].data.x[0])
    print(f"Node attribute dimension (in_dim): {in_dim} (expected: 169 = 118 + 51)")
    assert in_dim == 169, f"Expected in_dim=169 for CPD, got {in_dim}"
    
    em_dim = 64
    init_dict_cpd = dict(
        in_dim=118 + 51,
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
    
    dataloader = tg.loader.DataLoader(df_eval['data'].values, batch_size=900)
    
    # 7. Checkpoint inference loop over 100 checkpoints (100 to 199)
    folds = range(100, 200)
    n_structs = len(test_20)
    ensemble_preds = np.zeros((100, n_structs, out_dim), dtype=np.float64)
    
    print("\n--- Step 5: Running inference across 100 CPD checkpoints ---")
    loop_start = time.time()
    for idx_k, k in enumerate(folds):
        name = f"model_cpd_{k}.pt"
        run_name = f"CPD/{name}"
        assert os.path.exists(run_name), f"Checkpoint {run_name} not found!"
        
        model, opt, scheduler = get_model(init_dict_cpd, device=device)
        model.load_state_dict(torch.load(run_name, map_location=device))
        model.pool = True
        model.to(device)
        model.eval()
        
        with torch.no_grad():
            for d in dataloader:
                d.to(device)
                output = model(d)
                ensemble_preds[idx_k] = output.cpu().numpy()
                
        elapsed = time.time() - loop_start
        eta = (elapsed / (idx_k + 1)) * (100 - (idx_k + 1))
        print(f"[{idx_k+1:3d}/100] Checkpoint {name} completed. Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s", flush=True)
        
    # 8. Compute ensemble average and SAVE BEFORE ASSERTING
    pred_avg = np.mean(ensemble_preds, axis=0)
    out_npy = "cpd_port_check_ensemble_20.npy"
    np.save(out_npy, ensemble_preds)
    print(f"\nSaved ensemble predictions array to {out_npy} (shape {ensemble_preds.shape}).")
    
    # 9. Compare with published CPD.json predictions
    print("\n--- Step 6: Comparison with Published CPD.json Spectra ---")
    print(f"{'ID':<6} {'Formula':<12} {'Max Abs Diff':<14} {'Mean Abs Diff':<14} {'np.allclose':<12}")
    print("-" * 62)
    
    mismatches = 0
    all_close_flags = []
    for idx, tid in enumerate(test_20):
        pub_spectrum = np.array(cpd_pub['pred_avg'][str(tid)])
        our_spectrum = pred_avg[idx]
        
        max_diff = np.max(np.abs(our_spectrum - pub_spectrum))
        mean_diff = np.mean(np.abs(our_spectrum - pub_spectrum))
        is_close = np.allclose(our_spectrum, pub_spectrum, rtol=1e-3, atol=1e-5)
        all_close_flags.append(is_close)
        
        formula = orig_df.loc[int(tid), 'comp']
        print(f"{tid:<6} {formula:<12} {max_diff:<14.6f} {mean_diff:<14.6f} {str(is_close):<12}")
        if not is_close:
            mismatches += 1
            
    print("-" * 62)
    print(f"Summary: {20 - mismatches}/20 structures passed np.allclose(rtol=1e-3, atol=1e-5).")
    total_elapsed = time.time() - start_time
    print(f"Total time elapsed: {total_elapsed:.2f}s.")
    
    assert mismatches == 0, f"CPD Port Check FAILED: {mismatches}/20 structures failed np.allclose(rtol=1e-3, atol=1e-5)!"
    print("CPD Port Check PASSED successfully!")

if __name__ == '__main__':
    main()
