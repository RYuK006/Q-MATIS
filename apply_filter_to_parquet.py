import pandas as pd
from pymatgen.core import Composition

df = pd.read_parquet("novel_candidates_results.parquet")

EXCLUSION_LIST = set([
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",
    "Tl", "Hg", "Cd", "Os", "Ir"
])

def contains_excluded(formula):
    comp = Composition(formula)
    return any(el.symbol in EXCLUSION_LIST for el in comp.elements)

df['excluded_impractical'] = df['formula'].apply(contains_excluded)

print(f"Total candidates: {len(df)}")
print(f"Excluded: {df['excluded_impractical'].sum()}")
print(f"Practical: {len(df) - df['excluded_impractical'].sum()}")

df.to_parquet("novel_candidates_results.parquet")
print("Saved updated parquet with 'excluded_impractical' column.")
