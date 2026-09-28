import time
import os
import sys
import torch
import pandas as pd
import numpy as np
from pymatgen.core import Structure
from collections import defaultdict

structs_path = "wbm-init-structs.json.bz2"
summary_path = "wbm-summary.csv.gz"

df_summary = pd.read_csv(summary_path).set_index('material_id')
print("Loading structures...")
df_structs = pd.read_json(structs_path)
if 'material_id' in df_structs.columns:
    df_structs = df_structs.set_index('material_id')

struct_col = 'initial_structure'
df = df_structs.join(df_summary).dropna(subset=['e_above_hull_mp2020_corrected_ppd_mp'])

print(f"Total merged WBM size: {len(df)}")
df_sample = df.sample(n=1000, random_state=42).copy()
print(f"Sample size: {len(df_sample)}")

try:
    from mace.calculators import mace_mp
    macemp = mace_mp(model="small", device="cpu", default_dtype="float32")
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor
    
    print("Starting MACE-MP relaxation...")
    converged_count = 0
    times_by_bucket = defaultdict(list)
    
    with open("mace_results.csv", "w") as f:
        f.write("material_id,e_total,e_per_atom,n_atoms,formula,e_above_hull_true,e_form_wbm\n")
        
    for i, (idx, row) in enumerate(df_sample.iterrows()):
        struct_dict = row[struct_col]
        struct = Structure.from_dict(struct_dict) if isinstance(struct_dict, dict) else struct_dict
        n_atoms = len(struct)
        bucket = "<=4" if n_atoms <= 4 else ("5-10" if n_atoms <= 10 else ("11-20" if n_atoms <= 20 else "21-40"))
        
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = macemp
        
        t0 = time.time()
        try:
            opt = FIRE(FrechetCellFilter(atoms), logfile=None)
            opt.run(fmax=0.05, steps=500)
            t1 = time.time()
            times_by_bucket[bucket].append(t1 - t0)
            
            e_total = atoms.get_potential_energy()
            e_per_atom = e_total / n_atoms
            with open("mace_results.csv", "a") as f:
                f.write(f"{idx},{e_total},{e_per_atom},{n_atoms},{struct.composition.reduced_formula},{row['e_above_hull_mp2020_corrected_ppd_mp']},{row['e_form_per_atom_mp2020_corrected']}\n")
            converged_count += 1
        except Exception as e:
            pass
            
        if (i+1) % 10 == 0:
            print(f"Done {i+1}/1000", flush=True)
            
    print(f"\nMACE-MP Fraction converged: {converged_count / len(df_sample)}")
    for bucket, times in sorted(times_by_bucket.items(), key=lambda x: int(x[0].split('-')[0].replace('<=', '0'))):
        if times:
            avg_time = np.mean(times)
            print(f"MACE Bucket {bucket}: avg {avg_time:.3f}s -> {3600 / avg_time:.1f} relax/hr")

except Exception as e:
    print("MACE-MP failed:", e)
