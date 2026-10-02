import pandas as pd
import re

log_path = r'C:\Users\Aaron\.gemini\antigravity-ide\brain\d573a705-8fd8-4716-957e-36dc59b4d855\.system_generated\tasks\task-1635.log'
with open(log_path, 'r') as f:
    log_content = f.read()

ids_all = set(re.findall(r'ID = (mp-[a-zA-Z0-9\-]+)', log_content))

df = pd.read_csv('phonon_stability_results.csv')
ids_processed = set(df['material_id'].tolist())

missing_ids = ids_all - ids_processed

df_mp = pd.read_parquet('mp_reference_hull_with_structs.parquet')
df_missing = df_mp[df_mp['material_id'].isin(missing_ids)]
print('MISSING CANDIDATES:')
for _, r in df_missing.iterrows():
    print(f"{r['material_id']} | {r['formula']}")

df_stable = df[df['status'] == 'STABLE'].sort_values('min_freq_thz')
print('\nMARGIN LIST (STABLE):')
print(df_stable[['material_id', 'formula', 'min_freq_thz']].to_string(index=False))

total = 73
comfortably_stable = len(df_stable[df_stable['min_freq_thz'] > 0.2])
near_threshold = len(df_stable[df_stable['min_freq_thz'] <= 0.2])
unstable = len(df[df['status'] == 'UNSTABLE'])
skipped = len(df[df['status'] == 'SKIPPED_TOO_LARGE'])
inconclusive = len(missing_ids)

print('\nFUNNEL:')
print(f'Comfortably stable (>0.2 THz): {comfortably_stable} / {total} ({comfortably_stable/total*100:.1f}%)')
print(f'Near-threshold (<=0.2 THz): {near_threshold} / {total} ({near_threshold/total*100:.1f}%)')
print(f'Confirmed unstable (<0 THz): {unstable} / {total} ({unstable/total*100:.1f}%)')
print(f'Inconclusive (non-converging LBFGS): {inconclusive} / {total} ({inconclusive/total*100:.1f}%)')
print(f'Never evaluated (supercell >=150 atoms): {skipped} / {total} ({skipped/total*100:.1f}%)')
