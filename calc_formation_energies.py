import torch
import pandas as pd
import json
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

# Setup MACE
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading MACE on {device}...")
macemp = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)

def relax_structure(st):
    atoms = AseAtomsAdaptor.get_atoms(st)
    atoms.calc = macemp
    cell_filter = FrechetCellFilter(atoms)
    opt = FIRE(cell_filter, logfile=None)
    opt.run(fmax=0.05, steps=500)
    return atoms.get_potential_energy(), len(atoms)

print("Loading candidates...")
df_cand = pd.read_parquet("novel_candidates_results.parquet")
unique_elements = set(["Na", "Cl"])
for form in df_cand['formula']:
    comp = Composition(form)
    for el in comp.elements:
        unique_elements.add(el.symbol)

print(f"Found {len(unique_elements)} unique elements to establish baselines for.")

print("Finding elemental reference structures from MP...")
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

element_refs = {}
for el in unique_elements:
    def is_pure_element(form, target_el):
        try:
            comp = Composition(form)
            return len(comp.elements) == 1 and list(comp.elements)[0].symbol == target_el
        except:
            return False

    el_df = df_mp[df_mp['formula'].apply(lambda f: is_pure_element(f, el))]
    stable_el_df = el_df[el_df['e_hull'] <= 1e-4]
    
    if stable_el_df.empty:
        stable_el_df = el_df.sort_values('uncorrected_energy_per_atom')
        
    if stable_el_df.empty:
        print(f"WARNING: Could not find pure element phase for {el} in df_mp!")
        continue
        
    best_row = stable_el_df.iloc[0]
    st = Structure.from_dict(json.loads(best_row['structure_dict']))
    
    # Relax with MACE
    e_tot, n_atoms = relax_structure(st)
    e_per_atom = e_tot / n_atoms
    element_refs[el] = e_per_atom
    print(f"  {el:2s} -> {e_per_atom:9.4f} eV/atom (from {best_row['formula']})")

print("\nRunning NaCl Sanity Check...")
nacl_df = df_mp[df_mp['formula'] == 'NaCl'].sort_values('e_hull')
nacl_st = Structure.from_dict(json.loads(nacl_df.iloc[0]['structure_dict']))
e_tot_nacl, n_nacl = relax_structure(nacl_st)
comp_nacl = nacl_st.composition
ref_sum_nacl = sum(comp_nacl[el] * element_refs[el.symbol] for el in comp_nacl.elements)
e_form_nacl = (e_tot_nacl - ref_sum_nacl) / n_nacl
print(f"NaCl MACE formation energy: {e_form_nacl:.4f} eV/atom (Expected: ~ -2.1 eV/atom)")

print("\nComputing MACE formation energies for all 512 candidates...")
formation_energies = []
for _, row in df_cand.iterrows():
    comp = Composition(row['formula'])
    n_atoms = comp.num_atoms
    e_tot = row['raw_energy_per_atom'] * n_atoms
    
    ref_sum = 0
    for el in comp.elements:
        ref_sum += comp[el] * element_refs[el.symbol]
        
    e_form = (e_tot - ref_sum) / n_atoms
    formation_energies.append(e_form)

df_cand['mace_formation_energy'] = formation_energies

print("\n=== MACE FORMATION ENERGY DISTRIBUTION (relative to elements) ===")
print(f"Min:    {df_cand['mace_formation_energy'].min():.4f} eV/atom")
print(f"Median: {df_cand['mace_formation_energy'].median():.4f} eV/atom")
print(f"Max:    {df_cand['mace_formation_energy'].max():.4f} eV/atom")

# Also print Top 5 just to see
top5 = df_cand.sort_values('mace_formation_energy').head(5)
print("\nTop 5 Candidates by Formation Energy:")
print(top5[['formula', 'mace_formation_energy']].to_string(index=False))

df_cand.to_parquet("novel_candidates_results.parquet")
print("\nSaved updated results with mace_formation_energy back to parquet.")
