import torch
import pandas as pd
import json
from pymatgen.core import Structure, Composition
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

print("Loading MP reference hull...")
df_mp = pd.read_parquet('mp_reference_hull_with_structs.parquet')

# Need terminal elements for PD? The old script did:
terminal_elements = set()
for _, row in df_mp.iterrows():
    comp = Composition(row['formula'])
    if len(comp.elements) == 1:
        terminal_elements.add(list(comp.elements)[0].symbol)
        
raw_entries = []
for _, row in df_mp.iterrows():
    raw_entries.append(PDEntry(row['formula'], row['uncorrected_energy_per_atom'] * row['n_atoms']))

pd_entries = [e for e in raw_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
macemp = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)

# Find NaCl in df_mp
print("Finding NaCl...")
nacl_df = df_mp[df_mp['formula'] == 'NaCl'].sort_values(by='e_hull')
if nacl_df.empty:
    nacl_df = df_mp[df_mp['formula'] == 'Si'] # fallback
nacl_row = nacl_df.iloc[0]
nacl_st = Structure.from_dict(json.loads(nacl_row['structure_dict']))
print(f"Testing on {nacl_row['formula']} (MP e_hull: {nacl_row['e_hull']} eV/atom)")

atoms = AseAtomsAdaptor.get_atoms(nacl_st)
atoms.calc = macemp
cell_filter = FrechetCellFilter(atoms)
opt = FIRE(cell_filter, logfile='-')
opt.run(fmax=0.05, steps=500)

e_total = atoms.get_potential_energy()
comp = nacl_st.composition

system_elements = set(el.symbol for el in comp.elements)
local_entries = [e for e in pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)]
local_entries.append(PDEntry(comp, e_total))
pd_local = PhaseDiagram(local_entries)
new_entry = PDEntry(comp, e_total)
e_hull = pd_local.get_e_above_hull(new_entry)

print(f"\nFinal {comp.reduced_formula} MACE e_total: {e_total} eV")
print(f"Calculated MACE E_hull result: {e_hull} eV/atom")
