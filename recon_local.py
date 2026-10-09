import json
import pandas as pd
import os
import sys
import numpy as np



print("=== 1. database.json info ===")
df = pd.read_json('https://raw.githubusercontent.com/henniggroup/BETE-NET/main/database.json')
print("Columns:", df.columns.tolist())
print("First 3 row labels:", df.index[:3].tolist())
if 'index' in df.columns:
    print("First 3 values of 'index' column:", df['index'].head(3).tolist())
else:
    print("'index' column not found in database.json")

import urllib.request
idx_test_url = "https://raw.githubusercontent.com/henniggroup/BETE-NET/main/indices/idx_test_full.txt"
content = urllib.request.urlopen(idx_test_url).read().decode('utf-8')
test_ids = []
for line in content.splitlines():
        val = line.strip()
        if val:
            try: test_ids.append(str(int(float(val))))
            except: test_ids.append(val)

print("\nState whether idx_test_full.txt values are row labels, positions or structure ids:")
print("Looking at first test id:", test_ids[0])
print("Is it in df.index?", int(test_ids[0]) in df.index or str(test_ids[0]) in df.index)
print("Is it in df['index']?", test_ids[0] in df['index'].astype(str).values if 'index' in df.columns else "N/A")

print("\n=== ls structures | head ===")
os.system("ls structures | head")

print("\n=== Check first 5 test ids structures ===")
for tid in test_ids[:5]:
    cif_path = f"structures/{tid}.cif"
    exists = os.path.exists(cif_path)
    print(f"{cif_path} exists: {exists}")
    
print("\n=== 2. Comp check ===")
cso_df = pd.read_json("https://raw.githubusercontent.com/henniggroup/BETE-NET/main/test_preds/CSO.json")
for tid in test_ids[:5]:
    db_comp = df.loc[int(tid), 'comp'] if int(tid) in df.index else df.loc[tid, 'comp'] if tid in df.index else None
    cso_comp = cso_df.loc[str(tid), 'comp'] if str(tid) in cso_df.index else None
    print(f"ID {tid}: db_comp={db_comp}, cso_comp={cso_comp}")
    
print("\n=== 3. CSO.json a2F vs get_target ===")
sys.path.append(os.getcwd())
from notebooks.utils.data import get_target
tid = test_ids[0]
db_row = df.loc[int(tid)] if int(tid) in df.index else df.loc[tid]
target_dft = get_target(db_row)
cso_a2f = cso_df.loc[str(tid), 'a2F']
print(f"ID {tid} DFT target (first 5): {target_dft[:5]}")
print(f"ID {tid} CSO a2F (first 5): {cso_a2f[:5]}")
max_diff = np.max(np.abs(target_dft - cso_a2f))
print(f"Max abs diff: {max_diff}")

print("\n=== 4. Confirm candidates CIFs ===")
phonon_df = pd.read_csv('../phonon_stability_results.csv')
stable = phonon_df[phonon_df['status'] == 'STABLE']
stable_ids = [s for s in stable['material_id'] if s != 'mp-aaabfqpo']
print(f"ls structures/mp-*.cif | wc -l : {os.popen('ls structures/mp-*.cif 2>/dev/null | wc -l').read().strip()}")
missing = 0
for sid in stable_ids:
    if not os.path.exists(f"../cifs_for_colab/{sid}.cif") and not os.path.exists(f"../structures/{sid}.cif"):
        # Wait, the prompt says check structures/mp-*.cif in the Colab. We just have to write the code for the user or test it locally
        pass
