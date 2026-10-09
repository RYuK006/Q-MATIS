import os
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
mp_api_key = os.getenv("MP_API_KEY")

print("=== Fetching all stable MP materials (e_hull <= 1e-5) ===")
try:
    with MPRester(mp_api_key) as mpr:
        docs = mpr.materials.summary.search(
            energy_above_hull=(-1e-5, 1e-5),
            fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"]
        )
        
    df_api = pd.DataFrame([
        {
            "material_id": str(d.material_id),
            "formula": str(d.formula_pretty),
            "band_gap": getattr(d, 'band_gap', None),
            "energy_above_hull": getattr(d, 'energy_above_hull', 0.0)
        }
        for d in docs
    ])
    
    # 1. Report ambiguity
    formula_counts = df_api['formula'].value_counts()
    ambiguous = formula_counts[formula_counts > 1]
    print(f"\nFormulas mapping to >1 MP entry on the hull: {len(ambiguous)}")
    if len(ambiguous) > 0:
        print("Examples:")
        for f, count in ambiguous.head(5).items():
            print(f"  {f}: {count} entries")
            
    # 2. Fix the picking logic
    print("\nSorting by energy_above_hull and picking the most stable...")
    df_api_sorted = df_api.sort_values('energy_above_hull')
    bg_data_fixed = df_api_sorted.drop_duplicates(subset=['formula'], keep='first').set_index('formula')['band_gap'].to_dict()
    
    # Re-apply to the dedup_mp pool
    df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
    stable_mp = df_mp[df_mp['e_hull'] <= 1e-5]

    from pymatgen.core import Composition
    def get_element_set(formula):
        try: return frozenset([el.symbol for el in Composition(formula).elements])
        except: return frozenset()
            
    sc_df = pd.read_csv("data/supercon.csv")
    sc_element_sets = set(sc_df['name'].dropna().apply(get_element_set))
    stable_mp['element_set'] = stable_mp['formula'].apply(get_element_set)
    dedup_mp = stable_mp[~stable_mp['element_set'].isin(sc_element_sets)]
    
    dedup_mp['band_gap'] = dedup_mp['formula'].apply(lambda x: bg_data_fixed.get(str(x), None))
    has_bg = dedup_mp.dropna(subset=['band_gap'])
    metallic_mp = has_bg[has_bg['band_gap'] <= 0.1]
    
    print(f"\nRecalculated Pool Sizes (fixed picking):")
    print(f"Entries with band_gap data: {len(has_bg)}")
    print(f"Band gap == 0: {(has_bg['band_gap'] == 0).sum()}")
    print(f"Band gap > 0:  {(has_bg['band_gap'] > 0).sum()}")
    print(f"Pool size after band_gap <= 0.1 eV filter: {len(metallic_mp)}")
    
except Exception as e:
    print(f"Failed: {e}")
