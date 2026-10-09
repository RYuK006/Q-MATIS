import os
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json

# Create output directory
out_dir = "cifs_for_colab"
os.makedirs(out_dir, exist_ok=True)

# 1. External References
refs = {"Mo2CN": "mp-aaacrmzn", "HfZrN2": "mp-aaacrral", "BeAlB": "mp-aaaaaghy", "Nb8PtSe20": "mp-aaacpwsj"}

df_ref = pd.read_parquet('mp_reference_hull_with_structs.parquet')
df_ref = df_ref.set_index('material_id')

df_novel = pd.read_parquet('novel_candidates_results.parquet')

ph_stable = pd.read_csv('phonon_stability_results.csv')
stable_ids = ph_stable[ph_stable['status'] == 'STABLE']['material_id'].tolist()
print(f"Found {len(stable_ids)} stable candidates.")

for name, mpid in refs.items():
    if mpid in df_ref.index:
        struct_data = df_ref.loc[mpid, 'structure_dict']
        if isinstance(struct_data, str):
            struct_data = json.loads(struct_data)
        struct = Structure.from_dict(struct_data)
        struct.to(filename=f"{out_dir}/{mpid}.cif")
        print(f"Exported {name} ({mpid}) from df_ref")
        
        if mpid == "mp-aaacpwsj":
            sga = SpacegroupAnalyzer(struct)
            niggli_struct = struct.get_reduced_structure()
            niggli_struct.to(filename=f"{out_dir}/{mpid}_niggli.cif")
            print(f"Exported {name} Niggli ({mpid}_niggli)")

    elif mpid in df_novel.index:
        struct = df_novel.loc[mpid, 'relaxed_structure']
        struct.to(filename=f"{out_dir}/{mpid}.cif")
        print(f"Exported {name} ({mpid}) from df_novel")

for mpid in stable_ids:
    if mpid in df_ref.index:
        struct_data = df_ref.loc[mpid, 'structure_dict']
        if isinstance(struct_data, str):
            struct_data = json.loads(struct_data)
        struct = Structure.from_dict(struct_data)
        struct.to(filename=f"{out_dir}/{mpid}.cif")
    elif mpid in df_novel.index:
        struct = df_novel.loc[mpid, 'relaxed_structure']
        struct.to(filename=f"{out_dir}/{mpid}.cif")
    else:
        print(f"Missing structure for {mpid}")

print("Done exporting CIFs.")
