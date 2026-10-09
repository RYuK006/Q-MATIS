import numpy as np
import pandas as pd

df = pd.read_json('BETE-NET-zip/BETE-NET-main/database.json')

with open('BETE-NET-zip/BETE-NET-main/indices/idx_test_full.txt') as f:
    test_ids_raw = [line.strip() for line in f if line.strip()]

test_ids = [int(float(x)) for x in test_ids_raw if int(float(x)) in df.index]

print(f"Total held-out test IDs in database.json: {len(test_ids)}")

# Find hydrides in test_ids and in df
hydrides_test = [tid for tid in test_ids if 'H' in df.loc[tid, 'comp']]
print(f"Hydrides in test set: {len(hydrides_test)}")
for h in hydrides_test[:10]:
    print(f"  ID {h}: {df.loc[h, 'comp']}")

# Let's inspect test IDs with modes above 100 meV
# In database.json, check PhFreq_meV and Ph_2x2x2_interpolated_Freq_meV
above_100_orig = []
above_100_coarse = []

for tid in test_ids:
    row = df.loc[tid]
    # original dense frequencies
    f_orig = np.array(row['PhFreq_meV'])
    if np.max(f_orig) > 100.0:
        above_100_orig.append((tid, row['comp'], np.max(f_orig)))
    
    # coarse / interpolated
    f_coarse = np.array(row['Ph_2x2x2_interpolated_Freq_meV'])
    if np.max(f_coarse) > 100.0:
        above_100_coarse.append((tid, row['comp'], np.max(f_coarse)))

print(f"\nTest IDs with original PhFreq_meV max > 100 meV: {len(above_100_orig)} of {len(test_ids)}")
for tid, comp, fmax in above_100_orig:
    print(f"  ID {tid} ({comp}): max freq = {fmax:.2f} meV")

print(f"\nTest IDs with 2x2x2 PhFreq max > 100 meV: {len(above_100_coarse)} of {len(test_ids)}")
for tid, comp, fmax in above_100_coarse:
    print(f"  ID {tid} ({comp}): max freq = {fmax:.2f} meV")
