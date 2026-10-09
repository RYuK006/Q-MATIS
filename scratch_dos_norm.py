import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

df = pd.read_json('BETE-NET-zip/BETE-NET-main/database.json')

Freq_final = np.arange(0.25, 101, 2)

def process_single_dos(x_orig, y_orig):
    xl = np.arange(0.25, 101, 0.1)
    y_interp = np.interp(xl, x_orig, y_orig)
    Y_sg = savgol_filter(y_interp, 101, 3, mode="interp")
    Y_grid = np.interp(Freq_final, xl, Y_sg)
    Y_grid = np.asarray([y if y > 0.0 else 0.0 for y in Y_grid])
    return Y_grid

# Pick 5 entries: 1 hydride (671: H2Cr), and 4 other test IDs (71: Bi4Rb2, 0: Cs, 148: Br2Cu, 297: Ag2F)
sample_ids = [671, 71, 0, 148, 297]

print(f"{'ID':<6} {'Comp':<10} {'Atom':<6} {'Orig Freq Max':<14} {'Orig Integral':<15} {'Processed 51pt':<15} {'% Retained':<10}")
print("-" * 80)

for sid in sample_ids:
    row = df.loc[sid]
    comp = row['comp']
    f_orig = np.array(row['Ph_2x2x2_interpolated_Freq_meV'])
    site_dos_list = row['Ph_2x2x2_interpolated_Site_Proj_DOS']
    
    for atom_idx, y_site in enumerate(site_dos_list):
        y_site = np.array(y_site)
        orig_int = np.trapezoid(y_site, f_orig) if hasattr(np, 'trapezoid') else np.trapz(y_site, f_orig)
        
        y_proc = process_single_dos(f_orig, y_site)
        proc_int = np.trapezoid(y_proc, Freq_final) if hasattr(np, 'trapezoid') else np.trapz(y_proc, Freq_final)
        
        pct = (proc_int / orig_int) * 100 if orig_int > 0 else 0
        print(f"{sid:<6} {comp:<10} Site {atom_idx:<3} {np.max(f_orig):<14.2f} {orig_int:<15.4f} {proc_int:<15.4f} {pct:<10.1f}%")
