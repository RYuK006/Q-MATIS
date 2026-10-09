import os
import sys
import json
import time
import numpy as np
import pandas as pd

def parse_id(raw_id):
    """Safely parse IDs like '7.100000000000000000e+01' to integer ID string '71'."""
    try:
        return str(int(float(raw_id.strip())))
    except Exception:
        return raw_id.strip()

def run_cpd_port_check_dryrun():
    print("=== [laptop] DRY RUN: cpd_port_check.py (STUBBED MODEL) ===")
    
    db_path = 'BETE-NET-zip/BETE-NET-main/database.json'
    idx_test_path = 'BETE-NET-zip/BETE-NET-main/indices/idx_test_full.txt'
    cpd_json_path = 'BETE-NET-zip/BETE-NET-main/test_preds/CPD.json'
    
    assert os.path.exists(db_path), f"database.json not found at {db_path}"
    assert os.path.exists(idx_test_path), f"idx_test_full.txt not found at {idx_test_path}"
    assert os.path.exists(cpd_json_path), f"CPD.json not found at {cpd_json_path}"
    
    # Load pristine database.json
    orig_df = pd.read_json(db_path)
    print(f"Loaded database.json: {len(orig_df)} rows.")
    
    # Load test indices
    with open(idx_test_path, 'r') as f:
        raw_test_lines = [line.strip() for line in f if line.strip()]
    
    # Parse test IDs with parse_id()
    parsed_test_ids = [parse_id(line) for line in raw_test_lines]
    print(f"Total raw test IDs in {idx_test_path}: {len(parsed_test_ids)}")
    
    # Filter to IDs present in database.json (integer index)
    test_held_out = [tid for tid in parsed_test_ids if int(tid) in orig_df.index]
    dropped_test_ids = [tid for tid in parsed_test_ids if int(tid) not in orig_df.index]
    print(f"Held-out test IDs in database.json: {len(test_held_out)}")
    print(f"Dropped test IDs: {dropped_test_ids} (expected: ['36', '786', '218'])")
    assert set(dropped_test_ids) == {'36', '786', '218'}, f"Unexpected dropped test IDs: {dropped_test_ids}"
    
    # Select first 20 held-out test IDs
    test_20 = test_held_out[:20]
    print(f"Selecting first 20 test IDs: {test_20}")
    assert len(test_20) == 20, f"Expected 20 test IDs, got {len(test_20)}"
    
    # Stream / load CPD.json to verify compositions and targets
    print("Loading CPD.json...")
    with open(cpd_json_path, 'r') as f:
        cpd_pub = json.load(f)
    
    print("Asserting composition matches between database.json and CPD.json...")
    for tid in test_20:
        db_comp = orig_df.loc[int(tid), 'comp']
        # CPD.json uses str(tid) as inner key
        assert str(tid) in cpd_pub['comp'], f"ID {tid} missing from CPD.json comp keys"
        cpd_comp = cpd_pub['comp'][str(tid)]
        assert db_comp == cpd_comp, f"Composition mismatch for ID {tid}: DB '{db_comp}' vs CPD '{cpd_comp}'"
    print("All 20 test compositions match CPD.json perfectly.")
    
    # Check structure files exist
    for tid in test_20:
        cif_p = f"BETE-NET-zip/BETE-NET-main/structures/{tid}.cif"
        assert os.path.exists(cif_p), f"Structure CIF {cif_p} missing!"
    print("All 20 structure CIFs verified present.")
    
    # Simulate 100 CPD checkpoints with stubbed model (random 51-vectors)
    print("Running stubbed inference (100 checkpoints x 20 structures x 51 frequencies)...")
    np.random.seed(42)
    n_checkpoints = 100
    n_structs = len(test_20)
    out_dim = 51
    ensemble_preds = np.zeros((n_checkpoints, n_structs, out_dim), dtype=np.float64)
    
    start_t = time.time()
    for k in range(n_checkpoints):
        # Stub: generate random output simulating model forward pass
        ensemble_preds[k] = np.random.uniform(0.0, 1.0, size=(n_structs, out_dim))
        if (k + 1) % 25 == 0 or k == 0:
            elapsed = time.time() - start_t
            eta = (elapsed / (k + 1)) * (n_checkpoints - (k + 1))
            print(f"  [{k+1:3d}/{n_checkpoints}] checkpoint simulated. Elapsed: {elapsed:.2f}s, ETA: {eta:.2f}s")
            
    # Compute ensemble average across 100 members
    pred_avg = np.mean(ensemble_preds, axis=0)
    assert pred_avg.shape == (20, 51), f"Unexpected shape for pred_avg: {pred_avg.shape}"
    
    # SAVE ENSEMBLE ARRAY BEFORE ASSERTING COMPARISON
    out_npy = "cpd_port_check_ensemble_20_stub.npy"
    np.save(out_npy, ensemble_preds)
    print(f"Saved stubbed ensemble predictions to {out_npy} (shape {ensemble_preds.shape}).")
    
    # Compare with published CPD.json predictions
    print("\nComparing with published CPD.json predictions (np.allclose rtol=1e-3, atol=1e-5):")
    mismatches = 0
    max_diffs = []
    for idx, tid in enumerate(test_20):
        pub_spectrum = np.array(cpd_pub['pred_avg'][str(tid)])
        our_spectrum = pred_avg[idx]
        is_close = np.allclose(our_spectrum, pub_spectrum, rtol=1e-3, atol=1e-5)
        max_abs = np.max(np.abs(our_spectrum - pub_spectrum))
        max_diffs.append(max_abs)
        if not is_close:
            mismatches += 1
            
    print(f"Stubbed model: {mismatches}/20 mismatches as expected (random vector vs published). Max diff = {np.max(max_diffs):.4f}")
    print("End-to-end data pipeline, ID indexing, composition matching, array saving verified successfully!")

if __name__ == '__main__':
    run_cpd_port_check_dryrun()
