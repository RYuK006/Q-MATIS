import os
import json
import time
import torch
import numpy as np
import pandas as pd
import ase.io
import torch_geometric as tg
from notebooks.utils.data import build_data, get_target, get_neighbors
from notebooks.utils.training import get_model

# 1. Compute train_num_neighbors from the original training database BEFORE overwriting it
print("Loading original database.json to compute train_num_neighbors...")
original_df = pd.read_json('database.json')
train_num_neighbors = get_neighbors(original_df, original_df.index).mean()
print(f"Corrected train_num_neighbors from training set: {train_num_neighbors:.4f}")

# 2. Overwrite database.json for our test candidates
test_candidates = ["mp-aaabfqpo", "mp-aaacpwsj"]
data_dicts = []
for idx in test_candidates:
    data_dicts.append({"index": idx, "target": np.zeros(51).tolist()})

df = pd.DataFrame(data_dicts)
df.to_json('database.json')
df.set_index('index', inplace=True)

# Generate CIF files if they don't exist
os.makedirs('structures', exist_ok=True)
with open('structures/mp-aaabfqpo.cif', 'w') as f:
    f.write("""# generated using pymatgen
data_Sr2Co2O5
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.43022300
_cell_length_b   5.55978839
_cell_length_c   8.94352183
_cell_angle_alpha   107.49651079
_cell_angle_beta   107.20983573
_cell_angle_gamma   91.47386402
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Sr2Co2O5
_chemical_formula_sum   'Sr4 Co4 O10'
_cell_volume   244.03447781
_cell_formula_units_Z   2
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Sr  Sr0  1  0.37986700  0.87678500  0.77722100  1.0
  Sr  Sr1  1  0.10459900  0.60085900  0.22303300  1.0
  Sr  Sr2  1  0.87913400  0.40000500  0.77734200  1.0
  Sr  Sr3  1  0.60247500  0.12364000  0.22239800  1.0
  Co  Co4  1  0.27631300  0.18432800  0.49796900  1.0
  Co  Co5  1  0.98849000  0.00012500  0.99979600  1.0
  Co  Co6  1  0.77609700  0.81479700  0.50171100  1.0
  Co  Co7  1  0.48932700  0.50089400  0.00088500  1.0
  O  O8  1  0.33059100  0.39288200  0.70838500  1.0
  O  O9  1  0.61881600  0.60723300  0.29009600  1.0
  O  O10  1  0.13032700  0.86524100  0.50004900  1.0
  O  O11  1  0.11926500  0.17345100  0.28155200  1.0
  O  O12  1  0.75172000  0.76131300  0.00876100  1.0
  O  O13  1  0.25385900  0.74719300  0.01002900  1.0
  O  O14  1  0.83751800  0.82905100  0.71966400  1.0
  O  O15  1  0.72704200  0.25721200  0.98986300  1.0
  O  O16  1  0.63015800  0.12893600  0.49974100  1.0
  O  O17  1  0.22840100  0.23605500  0.99150400  1.0
""")

with open('structures/mp-aaacpwsj.cif', 'w') as f:
    f.write("""# generated using pymatgen
data_Nb8PtSe20
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   10.58953195
_cell_length_b   10.58953195
_cell_length_c   19.71667769
_cell_angle_alpha   74.71110018
_cell_angle_beta   74.71110018
_cell_angle_gamma   18.96157369
_symmetry_Int_Tables_number   1
_chemical_formula_structural   Nb8PtSe20
_chemical_formula_sum   'Nb8 Pt1 Se20'
_cell_volume   692.27787812
_cell_formula_units_Z   1
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
  Nb  Nb0  1  0.07889791  0.07889791  0.09297451  1.0
  Nb  Nb1  1  0.92110209  0.92110209  0.90702549  1.0
  Nb  Nb2  1  0.32973698  0.32973698  0.81024832  1.0
  Nb  Nb3  1  0.67026302  0.67026302  0.18975168  1.0
  Nb  Nb4  1  0.26784964  0.26784964  0.28051142  1.0
  Nb  Nb5  1  0.73215036  0.73215036  0.71948858  1.0
  Nb  Nb6  1  0.15497015  0.15497015  0.55562591  1.0
  Nb  Nb7  1  0.84502985  0.84502985  0.44437409  1.0
  Pt  Pt8  1  0.50000000  0.50000000  0.00000000  1.0
  Se  Se9  1  0.44218766  0.44218766  0.39266541  1.0
  Se  Se10  1  0.55781234  0.55781234  0.60733459  1.0
  Se  Se11  1  0.40004108  0.40004108  0.51562733  1.0
  Se  Se12  1  0.59995892  0.59995892  0.48437267  1.0
  Se  Se13  1  0.14464987  0.14464987  0.69620423  1.0
  Se  Se14  1  0.85535013  0.85535013  0.30379577  1.0
  Se  Se15  1  0.38122717  0.38122717  0.00844620  1.0
  Se  Se16  1  0.61877283  0.61877283  0.99155380  1.0
  Se  Se17  1  0.20241139  0.20241139  0.82284142  1.0
  Se  Se18  1  0.79758861  0.79758861  0.17715858  1.0
  Se  Se19  1  0.25429748  0.25429748  0.42846082  1.0
  Se  Se20  1  0.74570252  0.74570252  0.57153918  1.0
  Se  Se21  1  0.32951795  0.32951795  0.67745971  1.0
  Se  Se22  1  0.67048205  0.67048205  0.32254029  1.0
  Se  Se23  1  0.48230586  0.48230586  0.12872417  1.0
  Se  Se24  1  0.51769414  0.51769414  0.87127583  1.0
  Se  Se25  1  0.20458300  0.20458300  0.08784978  1.0
  Se  Se26  1  0.79541700  0.79541700  0.91215022  1.0
  Se  Se27  1  0.07304785  0.07304785  0.22774782  1.0
  Se  Se28  1  0.92695215  0.92695215  0.77225218  1.0
""")

structures = []
for index, row in df.iterrows():
    structures.append(ase.io.read(f'structures/{index}.cif'))
df['structure'] = structures

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
start_time = time.time()

# Checkpoint system: load existing checkpoint if present
import pickle
checkpoint_file = "inference_checkpoint.pkl"
completed_models = 0
if os.path.exists(checkpoint_file):
    print("Resuming from checkpoint...")
    with open(checkpoint_file, 'rb') as f:
        df, completed_models = pickle.load(f)
        print(f"Resumed at model index {completed_models}")

for k in range(completed_models, 100):
    name = f"model_cso_{k}.pt"
    run_name = f'CSO/{name}'
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
            
    # Checkpoint after every model is processed
    with open(checkpoint_file, 'wb') as f:
        pickle.dump((df, k + 1), f)
        
print(f"Ensemble inference took {time.time() - start_time:.2f} seconds")

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

folds = range(100)
def get_avg(row):
    pred = np.zeros(51)
    for i in folds:
        pred += row[f'pred_{i}']
    return pred / len(folds)

# Suppress performance warnings from pandas during repeated inserts
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

df['pred_avg'] = df.apply(get_avg, axis=1)
df['lamb_pred'] = df.apply(lambda row: cal_lamb(Freq_final, row['pred_avg']), axis=1)
df['wlog_pred'] = df.apply(lambda row: cal_w_log(Freq_final, row['pred_avg'], row['lamb_pred']) / 0.08617, axis=1)
df['Tc_pred'] = df.apply(lambda row: cal_tc(row['lamb_pred'], row['wlog_pred']), axis=1)

print("\\n--- FINAL RESULTS ---")
for index, row in df.iterrows():
    print(f"Candidate {index}: lambda = {row['lamb_pred']:.3f}, w_log = {row['wlog_pred']:.2f} K, Tc = {row['Tc_pred']:.3f} K")
