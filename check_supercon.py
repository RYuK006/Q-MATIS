import pandas as pd
from pymatgen.core import Composition
import re

top20 = [
    "Nb8PtSe20", "Ba(CoSn4)2", "Sr(In4Rh)2", "Ti2AlRe", "Ga40SnMo8",
    "Ba8Si43Ni3", "Ca2ScP3Pt7", "BeAlB", "CeNiSnH", "Rb2CuF4",
    "EuSn3", "Ga40AgMo8", "Nb4NiS8", "HfZrN2", "Ga40GeMo8",
    "Sr2FeCuSO3", "LuGePd2", "Ba2YbRuO6", "Nd3Co4Sn13", "Pr2BiO2"
]

def get_el_set(form):
    try: return set([el.symbol for el in Composition(form).elements])
    except: return set()

top20_sets = {f: get_el_set(f) for f in top20}

sc = pd.read_csv("data/supercon.csv")
# dropna and get sets
sc = sc.dropna(subset=['name'])
sc['el_set'] = sc['name'].apply(get_el_set)

print("Checking SuperCon for supersets/subsets...")
for f in top20:
    el_set = top20_sets[f]
    # Superset: SuperCon entry contains all elements of f, PLUS others
    supersets = sc[sc['el_set'].apply(lambda x: el_set.issubset(x) and len(x) > len(el_set))]
    # Subset: f contains all elements of SuperCon entry, PLUS others
    subsets = sc[sc['el_set'].apply(lambda x: x.issubset(el_set) and len(x) < len(el_set) and len(x) > 0)]
    
    if len(supersets) > 0 or len(subsets) > 0:
        print(f"\n[{f}] ({'-'.join(el_set)})")
        if len(supersets) > 0:
            print(f"  Supersets in SuperCon: {len(supersets)} entries. Examples: {', '.join(supersets['name'].head(3).tolist())}")
        if len(subsets) > 0:
            print(f"  Subsets in SuperCon: {len(subsets)} entries. Examples: {', '.join(subsets['name'].head(3).tolist())}")
            
