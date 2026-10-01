import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
import warnings

warnings.filterwarnings("ignore")

df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

print("Building Fixed Reference Phase Diagram...")
fixed_entries = []
terminal_elements = set()
for _, row in df_mp.iterrows():
    try:
        comp = Composition(row['formula'])
        energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
        fixed_entries.append(PDEntry(comp, energy))
        if len(comp.elements) == 1:
            terminal_elements.add(list(comp.elements)[0].symbol)
    except Exception:
        pass

fixed_pd_entries = [e for e in fixed_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]

def validate_entries(condition_df, name):
    test_cases = []
    for _, row in condition_df.iterrows():
        comp = Composition(row['formula'])
        if row['n_atoms'] > comp.num_atoms * 2: 
            test_cases.append(row)
        if len(test_cases) >= 5:
            break

    print(f"\n--- Validating E_hull for 5 {name} Compounds ---")
    for row in test_cases:
        comp = Composition(row['formula'])
        
        true_comp_energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
        test_entry = PDEntry(comp, true_comp_energy)
        
        system_elements = set(el.symbol for el in comp.elements)
        fixed_local_pd = PhaseDiagram([e for e in fixed_pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)])
        
        # NO DIVISION BY ATOM COUNT HERE!
        fixed_ehull = fixed_local_pd.get_e_above_hull(test_entry)
        
        print(f"{row['formula']:<15} | MP Reported E_hull: {row['e_hull'] * 1000:7.3f} meV/atom | Recomputed E_hull: {fixed_ehull * 1000:7.3f} meV/atom")

stable = df_mp[df_mp['e_hull'] <= 1e-6]
metastable = df_mp[(df_mp['e_hull'] >= 0.005) & (df_mp['e_hull'] <= 0.050)]

validate_entries(stable, "STABLE (E_hull == 0)")
validate_entries(metastable, "METASTABLE (5-50 meV/atom)")
