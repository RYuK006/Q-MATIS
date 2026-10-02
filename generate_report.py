import pandas as pd
import os

df = pd.read_csv('phonon_stability_results.csv')
df_stable = df[df['status'] == 'STABLE']
df_unstable = df[df['status'] == 'UNSTABLE']

report_path = r'C:\Users\Aaron\.gemini\antigravity-ide\brain\d573a705-8fd8-4716-957e-36dc59b4d855\phonon_stability_report.md'

with open(report_path, 'w') as f:
    f.write('# Phonon Stability Report (< 150 atoms)\n\n')
    
    f.write(f'## Dynamically Stable ({len(df_stable)} candidates)\n')
    f.write('| Material ID | Formula | Supercell Atoms | Min Freq (THz) |\n')
    f.write('|---|---|---|---|\n')
    for _, r in df_stable.iterrows():
        f.write(f"| {r['material_id']} | {r['formula']} | {int(r['supercell_atoms'])} | {r['min_freq_thz']:.3f} |\n")
        
    f.write('\n')
    
    f.write(f'## Dynamically Unstable ({len(df_unstable)} candidates)\n')
    f.write('| Material ID | Formula | Supercell Atoms | Min Freq (THz) | q-point |\n')
    f.write('|---|---|---|---|---|\n')
    for _, r in df_unstable.iterrows():
        f.write(f"| {r['material_id']} | {r['formula']} | {int(r['supercell_atoms'])} | {r['min_freq_thz']:.3f} | {r['qpoint']} |\n")
