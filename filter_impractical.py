import pandas as pd
from pymatgen.core import Composition

df = pd.read_parquet("novel_candidates_results.parquet")
original_count = len(df)

# Defined exclusion list (actinides, radioactive, plus Tl, Hg, Cd, Os, Ir)
# Actinides: Ac, Th, Pa, U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr
# Other radioactive/rare: Tc, Pm, Po, At, Rn, Fr, Ra
# Heavy metals requested to exclude: Tl, Hg, Cd, Os, Ir
exclusion_list = set([
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",
    "Tl", "Hg", "Cd", "Os", "Ir"
])

def contains_excluded(formula):
    comp = Composition(formula)
    return any(el.symbol in exclusion_list for el in comp.elements)

df['excluded'] = df['formula'].apply(contains_excluded)
excluded_count = df['excluded'].sum()

print(f"Total candidates: {original_count}")
print(f"Candidates containing excluded elements: {excluded_count}")

# Filter out
df_clean = df[~df['excluded']].copy()

print(f"\nRemaining synthesizable candidates: {len(df_clean)}")

# Show top 10 by mace_formation_energy
if 'mace_formation_energy' in df_clean.columns:
    df_clean = df_clean.sort_values('mace_formation_energy')
    print("\n--- TOP 10 SYNTHESIZABLE CANDIDATES (by MACE formation energy) ---")
    print(df_clean[['formula', 'prototype_id', 'mace_formation_energy']].head(10).to_string(index=False))
else:
    print("Warning: mace_formation_energy column not found in parquet.")
