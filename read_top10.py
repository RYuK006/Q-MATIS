import pandas as pd
df = pd.read_parquet("novel_candidates_results.parquet")
print("Top 10 candidates by corrected E_hull:\n")
print(df[['formula', 'prototype_id', 'raw_energy_per_atom', 'e_hull_pred_mace']].head(10).to_string(index=False))
