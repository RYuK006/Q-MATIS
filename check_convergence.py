import pandas as pd
import json
import torch
from ase.optimize import LBFGS
from mace.calculators import mace_mp
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

def main():
    df_mp = pd.read_parquet('mp_reference_hull_with_structs.parquet')
    row = df_mp[df_mp['material_id'] == 'mp-aaabfqpo'].iloc[0]
    
    pmg_struct = Structure.from_dict(json.loads(row['structure_dict']))
    ase_atoms = AseAtomsAdaptor.get_atoms(pmg_struct)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    calc = mace_mp(model="medium", dispersion=False, default_dtype="float64", device=device)
    ase_atoms.calc = calc
    
    print(f"Checking convergence for {row['material_id']} | {row['formula']} (primitive cell: {len(ase_atoms)} atoms)")
    opt = LBFGS(ase_atoms, logfile='-')
    opt.run(fmax=0.05, steps=500)
    
    print("Converged:", opt.converged())

if __name__ == "__main__":
    main()
