import json
import torch
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from phonopy import Phonopy
from mace.calculators import mace_mp
from ase.optimize import LBFGS
from phonopy.structure.atoms import PhonopyAtoms

def get_phonopy_atoms(ase_atoms):
    return PhonopyAtoms(
        symbols=ase_atoms.get_chemical_symbols(),
        cell=ase_atoms.get_cell(),
        scaled_positions=ase_atoms.get_scaled_positions()
    )

def main():
    target_ids = ['mp-aaaccfde', 'mp-aaacjkdg', 'mp-aaackyex', 'mp-aaahbbao']
    df = pd.read_parquet('mp_reference_hull_with_structs.parquet')
    df = df[df['material_id'].isin(target_ids)]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=device)
    
    for _, row in df.iterrows():
        mat_id = row['material_id']
        form = row['formula']
        
        pmg_struct = Structure.from_dict(json.loads(row['structure_dict']))
        ase_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
        
        # Calculate supercell
        a, b, c = ase_atoms.cell.lengths()
        sc_x = max(1, int(10 / a))
        sc_y = max(1, int(10 / b))
        sc_z = max(1, int(10 / c))
        
        print(f"[{mat_id} | {form}] Relaxing...", end=" ", flush=True)
        ase_atoms.calc = calc
        opt = LBFGS(ase_atoms, logfile=None)
        opt.run(fmax=0.05, steps=500)
        print("Done. Computing phonons...", end=" ", flush=True)
        
        sc_matrix = np.diag([sc_x, sc_y, sc_z])
        phonon = Phonopy(get_phonopy_atoms(ase_atoms), sc_matrix, primitive_matrix='P', symprec=1e-3, log_level=0)
        phonon.generate_displacements(distance=0.01)
        
        forces_sets = []
        for sc in phonon.supercells_with_displacements:
            if hasattr(sc, "get_chemical_symbols"):
                syms, pos, cll = sc.get_chemical_symbols(), sc.get_scaled_positions(), sc.get_cell()
            else:
                syms, pos, cll = sc.symbols, sc.scaled_positions, sc.cell
            ase_sc = AseAtomsAdaptor.get_atoms(Structure(cll, syms, pos))
            ase_sc.calc = calc
            forces_sets.append(ase_sc.get_forces())
            
        phonon.forces = forces_sets
        phonon.produce_force_constants()
        phonon.run_mesh([10, 10, 10])
        mesh_dict = phonon.get_mesh_dict()
        frequencies = mesh_dict['frequencies']
        min_freq = np.min(frequencies)
        
        if min_freq < -0.05: # Effectively 0 THz allowing for small acoustic modes
            min_idx = np.unravel_index(np.argmin(frequencies), frequencies.shape)
            qpoint = mesh_dict['qpoints'][min_idx[0]]
            print(f"UNSTABLE. Min freq = {min_freq:.3f} THz at q={qpoint}")
        else:
            print(f"STABLE. Min freq = {min_freq:.3f} THz")

if __name__ == "__main__":
    main()
