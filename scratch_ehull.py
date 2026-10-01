import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor
from mace.calculators import mace_mp
import torch

print("Loading MP reference hull...")
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

print("Parsing MP reference entries (WITH FIX)...")
raw_entries = []
terminal_elements = set()
for _, row in df_mp.iterrows():
    try:
        comp = Composition(row['formula'])
        energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
        raw_entries.append(PDEntry(comp, energy))
        if len(comp.elements) == 1:
            terminal_elements.add(list(comp.elements)[0].symbol)
    except Exception:
        pass

pd_entries = [e for e in raw_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]

df_res = pd.read_parquet("novel_candidates_results.parquet")
# The ones that had lowest (incorrect) e_hull: Y2Ge2S7, Sc2Ni2O7, Ho2Ni2O7
top3 = df_res[df_res['formula'].isin(['Y2Ge2S7', 'Sc2Ni2O7', 'Ho2Ni2O7'])]

macemp = mace_mp(model="medium", dispersion=False, default_dtype="float32", device='cuda' if torch.cuda.is_available() else 'cpu')

for _, cand in top3.iterrows():
    struct = cand['structure']
    atoms = AseAtomsAdaptor.get_atoms(struct)
    atoms.calc = macemp
    e_total = atoms.get_potential_energy()
    n_atoms = len(atoms)
    comp = struct.composition
    
    entry = PDEntry(comp, e_total)
    system_elements = set(el.symbol for el in comp.elements)
    local_entries = [e for e in pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)]
    local_pd = PhaseDiagram(local_entries)
    
    correct_e_hull = local_pd.get_e_above_hull(entry) / n_atoms
    print(f"\n--- {cand['formula']} ---")
    print(f"Incorrect E_hull (from file): {cand['e_hull_pred_mace']:.4f} eV/atom")
    print(f"Raw relaxed MACE energy: {e_total / n_atoms:.4f} eV/atom")
    print(f"Correct E_hull: {correct_e_hull * 1000:.2f} meV/atom")
