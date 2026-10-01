import pandas as pd
import json

df = pd.read_csv('mace_checkpoint.csv')

print('=== TOP 10 CANDIDATES ===')
top_10 = df.sort_values(by='e_hull_pred_mace').head(10)
print(top_10[['formula', 'e_hull_pred_mace', 'prototype_id', 'category']].to_string(index=False))

print('\n=== PROTOTYPE DIVERSITY ===')
print(df['prototype_id'].value_counts().to_string())

print('\n=== E_HULL DISTRIBUTION (eV/atom) ===')
print(f"Min:    {df['e_hull_pred_mace'].min():.6f}")
print(f"Median: {df['e_hull_pred_mace'].median():.6f}")
print(f"Max:    {df['e_hull_pred_mace'].max():.6f}")
