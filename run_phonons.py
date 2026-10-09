import pandas as pd
import numpy as np
import re
import os
import json
import torch
from ase import Atoms
from mace.calculators import mace_mp
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import LBFGS
from pymatgen.core import Structure

def get_phonopy_atoms(ase_atoms):
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.get_cell(),
        scaled_positions=ase_atoms.get_scaled_positions()
    )

def get_supercell_matrix(cell, target_length=10.0):
    lengths = np.linalg.norm(cell, axis=1)
    multipliers = np.ceil(target_length / lengths).astype(int)
    multipliers = np.clip(multipliers, 1, 3)
    return np.diag(multipliers)

def main():
    # 1. Parse IDs from the log
    log_path = r"C:\Users\Aaron\.gemini\antigravity-ide\brain\d573a705-8fd8-4716-957e-36dc59b4d855\.system_generated\tasks\task-1635.log"
    with open(log_path, 'r') as f:
        log_content = f.read()

    ids = set(re.findall(r"ID = (mp-[a-zA-Z0-9\-]+)", log_content))
    print(f"Parsed {len(ids)} unique candidate IDs from the top-20 lists.")

    # 2. Load structures from parquet
    print("Loading structures from mp_reference_hull_with_structs.parquet...")
    df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
    df_targets = df_mp[df_mp['material_id'].isin(ids)].copy()
    print(f"Found {len(df_targets)} matching structures in parquet.")

    print("Initializing MACE calculator...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=device)

    results_file = "phonon_stability_results.csv"
    if not os.path.exists(results_file):
        pd.DataFrame(columns=["material_id", "formula", "supercell_atoms", "min_freq_thz", "status", "qpoint"]).to_csv(results_file, index=False)

    completed_count = 0
    stable_count = 0
    unstable_count = 0

    print("\n=== Running Phonon Stability Checks ===")
    for i, row in df_targets.iterrows():
        mat_id = row['material_id']
        form = row['formula']
        
        existing = pd.read_csv(results_file)
        if mat_id in existing['material_id'].values:
            continue
            
        try:
            pmg_struct = Structure.from_dict(json.loads(row['structure_dict']))
            ase_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        except Exception as e:
            print(f"[{mat_id} | {form}] Error loading structure: {e}")
            continue
            
        sc_matrix = get_supercell_matrix(ase_atoms.get_cell())
        n_atoms_sc = len(ase_atoms) * np.prod(np.diag(sc_matrix))
        
        if n_atoms_sc >= 150:
            new_row = pd.DataFrame([{
                "material_id": mat_id, "formula": form, 
                "supercell_atoms": n_atoms_sc, "min_freq_thz": np.nan, 
                "status": "SKIPPED_TOO_LARGE", "qpoint": ""
            }])
            new_row.to_csv(results_file, mode='a', header=False, index=False)
            continue
            
        print(f"[{mat_id} | {form}] SC: {np.diag(sc_matrix)} ({n_atoms_sc} atoms). Relaxing...", end=" ", flush=True)
        
        try:
            ase_atoms.calc = calc
            opt = LBFGS(ase_atoms, logfile=None)
            opt.run(fmax=0.05, steps=500)
            
            print("Phonons...", end=" ", flush=True)
            phonon = Phonopy(get_phonopy_atoms(ase_atoms), sc_matrix, primitive_matrix='P', symprec=1e-3, log_level=0)
            phonon.generate_displacements(distance=0.01)
            
            forces_sets = []
            for sc in phonon.supercells_with_displacements:
                if hasattr(sc, "get_chemical_symbols"):
                    syms, pos, cll = sc.get_chemical_symbols(), sc.get_scaled_positions(), sc.get_cell()
                else:
                    syms, pos, cll = sc.symbols, sc.scaled_positions, sc.cell
                ase_sc = Atoms(symbols=syms, scaled_positions=pos, cell=cll, pbc=True)
                ase_sc.calc = calc
                forces_sets.append(ase_sc.get_forces())
                
            phonon.forces = forces_sets
            phonon.produce_force_constants()
            phonon.run_mesh([10, 10, 10])
            mesh_dict = phonon.get_mesh_dict()
            frequencies = mesh_dict['frequencies']
            
            min_freq = np.min(frequencies)
            
            if min_freq < -0.05: # Strict Threshold for imaginary freq (THZ)
                min_idx = np.unravel_index(np.argmin(frequencies), frequencies.shape)
                qpoint = mesh_dict['qpoints'][min_idx[0]]
                print(f"UNSTABLE. Min freq = {min_freq:.3f} THz at q={qpoint}")
                status = "UNSTABLE"
                qstr = str(qpoint)
                unstable_count += 1
            else:
                print(f"STABLE. Min freq = {min_freq:.3f} THz")
                status = "STABLE"
                qstr = ""
                stable_count += 1
                
            new_row = pd.DataFrame([{
                "material_id": mat_id, "formula": form, 
                "supercell_atoms": n_atoms_sc, "min_freq_thz": min_freq, 
                "status": status, "qpoint": qstr
            }])
            new_row.to_csv(results_file, mode='a', header=False, index=False)
            
            completed_count += 1
            if completed_count % 10 == 0:
                print(f"--- CHECKPOINT: {completed_count} <150-atom candidates processed ({stable_count} STABLE, {unstable_count} UNSTABLE) ---")
                
        except Exception as e:
            print(f"ERROR: {e}")
            new_row = pd.DataFrame([{
                "material_id": mat_id, "formula": form, 
                "supercell_atoms": n_atoms_sc, 
                "min_freq_thz": np.nan, "status": "ERROR", "qpoint": str(e)
            }])
            new_row.to_csv(results_file, mode='a', header=False, index=False)

    print("\n" + "="*50)
    print("=== PHONON STABILITY SUMMARY ===")
    print(f"Total <150-atom candidates evaluated: {completed_count}")
    print(f"Dynamically stable:   {stable_count}")
    print(f"Dynamically unstable: {unstable_count}")
    print("="*50)

if __name__ == "__main__":
    main()
