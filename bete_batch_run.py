import os
import json
import time
import torch
import numpy as np
import pandas as pd
import ase.io
import torch_geometric as tg
from notebooks.utils.data import build_data
from notebooks.utils.training import get_model

def main():
    start_time = time.time()
    
    # 1. Load the 5 test structures from BETE-NET's own test set
    print("=== 1. PORT CHECK ===")
    idx_test_path = "indices/idx_test_full.txt"
    test_ids = []
    if os.path.exists(idx_test_path):
        with open(idx_test_path, 'r') as f:
            test_ids = [line.strip() for line in f.readlines()[:5]]
    else:
        print(f"Error: {idx_test_path} not found.")
    
    ext_refs = {
        "mp-aaacrmzn": "Mo2CN", 
        "mp-aaacrral": "HfZrN2", 
        "mp-aaaaaghy": "BeAlB",
        "mp-aaacpwsj_niggli": "Nb8PtSe20 (Niggli)",
        "mp-aaacpwsj": "Nb8PtSe20 (Skewed)"
    }
    
    all_cifs = [f for f in os.listdir('structures') if f.endswith('.cif')]
    
    run_ids = test_ids + list(ext_refs.keys()) + [f.replace('.cif', '') for f in all_cifs if f.replace('.cif', '') not in test_ids and f.replace('.cif', '') not in ext_refs.keys()]
    run_ids = list(dict.fromkeys(run_ids)) # Remove duplicates while preserving order
    
    print(f"Total structures to process: {len(run_ids)}")
    
    # Generate database.json for these structures
    records = []
    for mpid in run_ids:
        # We need to create a dummy target to avoid errors in build_data
        records.append({"index": mpid, "target": np.zeros(51).tolist()})
        
    df = pd.DataFrame(records)
    df.to_json('database.json')
    df.set_index('index', inplace=True)
    
    structures = []
    for index, row in df.iterrows():
        try:
            structures.append(ase.io.read(f'structures/{index}.cif'))
        except Exception as e:
            print(f"Failed to read structures/{index}.cif: {e}")
            structures.append(None)
    df['structure'] = structures
    
    # Remove structures that failed to load
    df = df.dropna(subset=['structure'])
    
    print("Building data graphs...")
    # Fixed train_num_neighbors from training set
    train_num_neighbors = 14.75
    print(f"Using fixed train_num_neighbors = {train_num_neighbors:.4f}")
    
    df['data'] = df.apply(build_data, embed_ph_dos=False, embed_e_dos=False, fine=False, r_max=4, axis=1)
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dim = 51
    in_dim = len(df.iloc[0].data.x[0])
    em_dim = 64
    
    init_dict_base = dict(in_dim=118, em_dim=em_dim, irreps_in=str(em_dim)+"x0e", 
        irreps_out=str(out_dim)+"x0e", irreps_node_attr=str(em_dim)+"x0e", 
        layers=2, mul=32, lmax=1, max_radius=4, 
        num_neighbors=train_num_neighbors, reduce_output=True, p=0.0)
        
    dataloader = tg.loader.DataLoader(df['data'].values, batch_size=900)
    
    folds = range(1, 6) # Try running first 5 models to save time? User asked to run batch. Let's run 100 models like the paper! But wait, 100 models takes time. The notebook from before ran 100 models in 52 seconds!
    # Let's run all 100 models.
    folds = range(100)
    for k in folds:
        name = f"model_cso_{k}.pt"
        run_name = f'CSO/{name}'
        if not os.path.exists(run_name):
            continue
        model, opt, scheduler = get_model(init_dict_base, device=device)
        model.load_state_dict(torch.load(run_name, map_location=device))
        model.pool = True
        model.to(device)
        model.eval()
        
        df[f'pred_{k}'] = np.empty((len(df), 1)).tolist()
        with torch.no_grad():
            for i, d in enumerate(dataloader):
                d.to(device)
                output = model(d)
                df[f'pred_{k}'] = [val for val in output.cpu().numpy()]
                
    Freq_final = np.arange(0.25, 101, 2)
    def cal_lamb(freq_w, alpha_F):
        lambdaF = 0
        try:
            for i in range(1, len(freq_w)):
                dw = freq_w[i] - freq_w[i-1]
                w = freq_w[i]
                alpha_F_w = alpha_F[i]
                lambdaF = lambdaF + ((alpha_F_w/w)*dw)
            return 2*lambdaF
        except:
            return np.nan

    def cal_w_log(freq_w, alpha_F, lamb):
        w_logF = 0
        try:
            for i in range(1, len(freq_w)):
                dw = freq_w[i] - freq_w[i-1]
                w_logF = w_logF + (alpha_F[i]*np.log(freq_w[i])*dw/freq_w[i])
            return np.exp(2*w_logF/lamb)
        except:
            return np.nan

    def cal_tc(lamb, omega_log, mu=0.09):
        frac = -1.04*(1+lamb)/(lamb-mu*(1+0.62*lamb))
        return (omega_log/1.2)*np.exp(frac)
        
    def get_avg(row):
        pred = np.zeros(51)
        valid_folds = [i for i in folds if f'pred_{i}' in df.columns]
        for i in valid_folds:
            pred += row[f'pred_{i}']
        return pred / len(valid_folds) if valid_folds else pred
        
    df['pred_avg'] = df.apply(get_avg, axis=1)
    df['lamb_pred'] = df.apply(lambda row: cal_lamb(Freq_final, row['pred_avg']), axis=1)
    df['wlog_pred'] = df.apply(lambda row: cal_w_log(Freq_final, row['pred_avg'], row['lamb_pred']) / 0.08617, axis=1)
    df['Tc_pred'] = df.apply(lambda row: cal_tc(row['lamb_pred'], row['wlog_pred']), axis=1)
    
    print("\n--- TEST SET COMPARISON ---")
    df_test_ground = pd.read_json('test_preds/CSO.json') if os.path.exists('test_preds/CSO.json') else pd.DataFrame()
    for tid in test_ids:
        if tid in df.index:
            pred_row = df.loc[tid]
            if not df_test_ground.empty and tid in df_test_ground['index'].values:
                gt_row = df_test_ground[df_test_ground['index'] == tid].iloc[0]
                print(f"{tid}: Pred Tc={pred_row['Tc_pred']:.3f}, GT Tc={gt_row['pred']:.3f} | Pred w_log={pred_row['wlog_pred']:.3f}, GT w_log={gt_row['Freq_pred']:.3f}")
            else:
                print(f"{tid}: Pred Tc={pred_row['Tc_pred']:.3f} (GT not found)")
                
    print("\n--- EXTERNAL REFERENCES ---")
    for tid, name in ext_refs.items():
        if tid in df.index:
            pred_row = df.loc[tid]
            print(f"{name} ({tid}): Tc={pred_row['Tc_pred']:.3f} K, lambda={pred_row['lamb_pred']:.3f}, w_log={pred_row['wlog_pred']:.3f} K")
            
    print("\n--- ALL CANDIDATES ---")
    for index, row in df.iterrows():
        if index not in test_ids and index not in ext_refs.keys():
            print(f"{index}: Tc={row['Tc_pred']:.3f} K, lambda={row['lamb_pred']:.3f}, w_log={row['wlog_pred']:.3f} K")
            
    print(f"\nTotal runtime: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
