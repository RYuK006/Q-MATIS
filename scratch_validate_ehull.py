import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import warnings

# Suppress pymatgen warnings for clean output
warnings.filterwarnings("ignore")

df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

print("Building Fixed Reference Phase Diagram...")
fixed_entries = []
terminal_elements = set()
for _, row in df_mp.iterrows():
    try:
        comp = Composition(row['formula'])
        # FIX: Using comp.num_atoms instead of row['n_atoms']
        energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
        fixed_entries.append(PDEntry(comp, energy))
        if len(comp.elements) == 1:
            terminal_elements.add(list(comp.elements)[0].symbol)
    except Exception:
        pass

fixed_pd_entries = [e for e in fixed_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]

print("Building Buggy Reference Phase Diagram...")
buggy_entries = []
for _, row in df_mp.iterrows():
    try:
        comp = Composition(row['formula'])
        # BUG: Multiplying by the unit cell size
        energy = row['uncorrected_energy_per_atom'] * row['n_atoms']
        buggy_entries.append(PDEntry(comp, energy))
    except Exception:
        pass
buggy_pd_entries = [e for e in buggy_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]

# Find 5 stable entries where unit cell size != reduced formula size
stable = df_mp[df_mp['e_hull'] <= 1e-6]
test_cases = []
for _, row in stable.iterrows():
    comp = Composition(row['formula'])
    if row['n_atoms'] > comp.num_atoms * 2: # Find cases with significant unit cell vs reduced formula mismatch
        test_cases.append(row)
    if len(test_cases) >= 5:
        break

print("\n--- Validating E_hull for 5 Known Stable MP Compounds ---")
for row in test_cases:
    comp = Composition(row['formula'])
    
    # The energy calculated correctly for the composition
    true_comp_energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
    test_entry = PDEntry(comp, true_comp_energy)
    
    system_elements = set(el.symbol for el in comp.elements)
    
    # Local Fixed PD
    fixed_local_pd = PhaseDiagram([e for e in fixed_pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)])
    # Local Buggy PD
    buggy_local_pd = PhaseDiagram([e for e in buggy_pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)])
    
    fixed_ehull = fixed_local_pd.get_e_above_hull(test_entry) / comp.num_atoms
    buggy_ehull = buggy_local_pd.get_e_above_hull(test_entry) / comp.num_atoms
    
    print(f"\n{row['formula']}:")
    print(f"  Unit cell size: {row['n_atoms']} | Reduced formula size: {comp.num_atoms}")
    print(f"  E_hull (Fixed Logic): {fixed_ehull * 1000:7.3f} meV/atom")
    print(f"  E_hull (Buggy Logic): {buggy_ehull * 1000:7.3f} meV/atom")
