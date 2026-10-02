import pandas as pd
import numpy as np
import re
from ase import Atoms
from mace.calculators import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor
import json
import torch

# 1. Parse IDs from the log
log_path = r"C:\Users\Aaron\.gemini\antigravity-ide\brain\d573a705-8fd8-4716-957e-36dc59b4d855\.system_generated\tasks\task-1635.log"
with open(log_path, 'r') as f:
    log_content = f.read()

# Regex to find ID = mp-...
ids = set(re.findall(r"ID = (mp-[a-zA-Z0-9\-]+)", log_content))
print(f"Parsed {len(ids)} unique candidate IDs from the top-20 lists.")

# 2. Load structures from parquet
print("Loading structures from mp_reference_hull_with_structs.parquet...")
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
df_targets = df_mp[df_mp['material_id'].isin(ids)].copy()
print(f"Found {len(df_targets)} matching structures in parquet.")

def get_phonopy_atoms(ase_atoms):
    return PhonopyAtoms(symbols=ase_atoms.get_chemical_symbols(),
                        cell=ase_atoms.get_cell(),
                        scaled_positions=ase_atoms.get_scaled_positions())

def get_supercell_matrix(cell, target_length=12.0):
    # Ensure supercell dimensions are at least target_length Angstroms
    lengths = np.linalg.norm(cell, axis=1)
    multipliers = np.ceil(target_length / lengths).astype(int)
    # Cap at [3,3,3] to prevent OOM
    multipliers = np.clip(multipliers, 1, 3)
    return np.diag(multipliers)

print("Initializing MACE calculator...")
# We use cuda if available, otherwise cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
calc = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)

results = []
unstable_count = 0
stable_count = 0

print("\n=== Running Phonon Stability Checks ===")
for i, row in df_targets.iterrows():
    mat_id = row['material_id']
    form = row['formula']
    
    # pymatgen structure from dict
    try:
        from pymatgen.core import Structure
        pmg_struct = Structure.from_dict(json.loads(row['structure_dict']))
        ase_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    except Exception as e:
        print(f"[{mat_id} | {form}] Error loading structure: {e}")
        continue
        
    sc_matrix = get_supercell_matrix(ase_atoms.get_cell())
    n_atoms_sc = len(ase_atoms) * np.prod(np.diag(sc_matrix))
    print(f"[{mat_id} | {form}] Supercell: {np.diag(sc_matrix)} ({n_atoms_sc} atoms) -> ", end="", flush=True)
    
    if n_atoms_sc > 500:
        print("SKIPPED (Supercell too large > 500 atoms, likely complex unit cell)")
        continue
        
    try:
        
        phonon = Phonopy(get_phonopy_atoms(ase_atoms), sc_matrix, primitive_matrix='P', symprec=1e-3, log_level=0)
        phonon.generate_displacements(distance=0.01)
        supercells = phonon.supercells_with_displacements
        
        forces_sets = []
        for sc in supercells:
            if hasattr(sc, "get_chemical_symbols"):
                syms, pos, cll = sc.get_chemical_symbols(), sc.get_scaled_positions(), sc.get_cell()
            else:
                syms, pos, cll = sc.symbols, sc.scaled_positions, sc.cell
            ase_sc = Atoms(symbols=syms, scaled_positions=pos, cell=cll, pbc=True)
            ase_sc.calc = calc
            forces_sets.append(ase_sc.get_forces())
            
        phonon.produce_force_constants(forces=forces_sets)
        phonon.run_mesh([10, 10, 10])
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies'] # shape (qpoints, bands)
        
        min_freq = np.min(frequencies)
        
        if min_freq < -0.25: # Threshold for imaginary freq (THZ)
            # Find the q-point with minimum frequency
            min_idx = np.unravel_index(np.argmin(frequencies), frequencies.shape)
            qpoint = mesh_dict['qpoints'][min_idx[0]]
            print(f"UNSTABLE. Min freq = {min_freq:.2f} THz at q={qpoint}")
            unstable_count += 1
        else:
            if min_freq < 0:
                print(f"STABLE (Marginal acoustic mode artifact). Min freq = {min_freq:.3f} THz")
            else:
                print(f"STABLE. Min freq = {min_freq:.3f} THz")
            stable_count += 1
            
    except Exception as e:
        print(f"ERROR: {e}")

print("\n" + "="*50)
print("=== PHONON STABILITY SUMMARY ===")
print(f"Total candidates evaluated: {stable_count + unstable_count}")
print(f"Dynamically stable:   {stable_count}")
print(f"Dynamically unstable: {unstable_count}")
print("="*50)
