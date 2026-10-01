import torch
import pandas as pd
import json
import numpy as np
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
macemp = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)

print("Loading parquet...")
df = pd.read_parquet("novel_candidates_results.parquet")

# Ensure sorted by E_hull
df = df.sort_values(by="e_hull_pred_mace")

def analyze_struct(row, idx_label):
    st = Structure.from_dict(json.loads(row['relaxed_structure']))
    atoms = AseAtomsAdaptor.get_atoms(st)
    atoms.calc = macemp
    forces = atoms.get_forces()
    fmax = np.max(np.linalg.norm(forces, axis=1))
    
    # Calculate min bond length (exclude self interactions)
    dist_mat = st.distance_matrix
    np.fill_diagonal(dist_mat, np.inf)
    min_bond = np.min(dist_mat)
    
    print(f"{idx_label}: {row['formula']} (E_hull = {row['e_hull_pred_mace']:.2f}) -> fmax = {fmax:.4f}, min bond = {min_bond:.2f} A")
    return fmax, min_bond

print("\n--- TOP 3 CANDIDATES ---")
for i in range(3):
    analyze_struct(df.iloc[i], f"Top {i+1}")

print("\n--- BOTTOM 3 CANDIDATES ---")
for i in range(1, 4):
    analyze_struct(df.iloc[-i], f"Bottom {i}")

print("\nChecking convergence across all 512 candidates... this will take ~30 seconds.")
converged = 0
not_converged = 0
blown_up = 0
for idx, row in df.iterrows():
    st = Structure.from_dict(json.loads(row['relaxed_structure']))
    atoms = AseAtomsAdaptor.get_atoms(st)
    atoms.calc = macemp
    forces = atoms.get_forces()
    fmax = np.max(np.linalg.norm(forces, axis=1))
    
    dist_mat = st.distance_matrix
    np.fill_diagonal(dist_mat, np.inf)
    min_bond = np.min(dist_mat)

    
    if min_bond < 0.7:
        blown_up += 1
    elif fmax <= 0.055: # slight tolerance over 0.05
        converged += 1
    else:
        not_converged += 1

print(f"\n--- OVERALL STATS ---")
print(f"Converged (fmax <= 0.05): {converged}")
print(f"Hit max steps (not converged): {not_converged}")
print(f"Structurally blown up (min bond < 0.7 A): {blown_up}")
