import os
import pandas as pd
import numpy as np
from mp_api.client import MPRester
from dotenv import load_dotenv
import joblib
from superconductor.features import get_node_features
from pymatgen.core import Composition
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
mp_api_key = os.getenv("MP_API_KEY")

print("=== Fetching all stable MP materials (e_hull <= 1e-5) ===")
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

# Sort and pick the most stable (lowest energy_above_hull) to fix ambiguity
df_api_sorted = df_api.sort_values('energy_above_hull')
bg_data_fixed = df_api_sorted.drop_duplicates(subset=['formula'], keep='first').set_index('formula')['band_gap'].to_dict()

# Re-apply to the pool
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
stable_mp = df_mp[df_mp['e_hull'] <= 1e-5]

def get_element_set(formula):
    try: return frozenset([el.symbol for el in Composition(formula).elements])
    except: return frozenset()
        
EXCLUSION_LIST = set([
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",
    "Tl", "Hg", "Cd", "Os", "Ir"
])

sc_df = pd.read_csv("data/supercon.csv")
sc_element_sets = set(sc_df['name'].dropna().apply(get_element_set))
stable_mp['element_set'] = stable_mp['formula'].apply(get_element_set)

def has_excluded(el_set):
    return any(el in EXCLUSION_LIST for el in el_set)

# Dedup and exclude
dedup_mp = stable_mp[
    (~stable_mp['element_set'].isin(sc_element_sets)) & 
    (~stable_mp['element_set'].apply(has_excluded))
]

dedup_mp['band_gap'] = dedup_mp['formula'].apply(lambda x: bg_data_fixed.get(str(x), None))
has_bg = dedup_mp.dropna(subset=['band_gap'])
metallic_mp = has_bg[has_bg['band_gap'] <= 0.1].copy()

print(f"Metallic Pool Size after EXCLUSION_LIST: {len(metallic_mp)}")

print("\n=== Running ML Inference on Metallic Pool ===")
def get_composition_features(formula):
    try:
        comp = Composition(formula).fractional_composition
        features = []
        weights = []
        for el, amt in comp.items():
            class MockSite:
                def __init__(self, e):
                    self.specie = e
            features.append(get_node_features(MockSite(el)))
            weights.append(amt)
        features = np.array(features)
        weights = np.array(weights).reshape(-1, 1)
        mean_feat = np.sum(features * weights, axis=0)
        return mean_feat
    except Exception:
        return None

model = joblib.load("models/rf_tc_model.joblib")

results = []
failed = 0

# 1. Featurize all
formulas = metallic_mp['formula'].tolist()
material_ids = metallic_mp['material_id'].tolist()
features_list = []
valid_indices = []

print("Featurizing 13,410 candidates...")
for i, formula in enumerate(formulas):
    feat = get_composition_features(formula)
    if feat is not None:
        features_list.append(feat)
        valid_indices.append(i)
    else:
        failed += 1

print(f"Failed to parse formulas: {failed}")

# 2. Batch predict
if features_list:
    print("Running batched ML predictions...")
    X = np.vstack(features_list)
    pred_tcs = model.predict(X)
    
    if hasattr(model, "estimators_"):
        # Batch predict all trees
        print("Running batched uncertainty calculations...")
        # To avoid massive memory spike if X is huge, we can still list comprehension the trees, but over X!
        tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
        pred_stds = np.std(tree_preds, axis=0)
    else:
        pred_stds = np.zeros(len(X))
        
MAGNETIC_RISK_LIST = set([
    "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", 
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"
])

def has_mag_risk(form):
    el_set = get_element_set(form)
    return any(el in MAGNETIC_RISK_LIST for el in el_set)

for idx_in_valid, idx_in_full in enumerate(valid_indices):
    tc = pred_tcs[idx_in_valid]
    unc = pred_stds[idx_in_valid]
    ratio = unc / tc if tc > 0 else np.inf
    f = formulas[idx_in_full]
    results.append({
        'formula': f,
        'tc': tc,
        'unc': unc,
        'ratio': ratio,
        'material_id': material_ids[idx_in_full],
        'contains_magnetic_risk_element': has_mag_risk(f)
    })

df_results = pd.DataFrame(results)

# Fractions
pool_frac = df_results['contains_magnetic_risk_element'].mean()
print(f"\nFraction of 10,147 pool with magnetic risk: {pool_frac:.1%} ({df_results['contains_magnetic_risk_element'].sum()} candidates)")

top20_raw = df_results.sort_values(by='tc', ascending=False).head(20)
top20_ratio = df_results[df_results['tc'] > 1.0].sort_values(by='ratio', ascending=True).head(20)

print(f"Fraction of current Top 20 Raw Tc with magnetic risk: {top20_raw['contains_magnetic_risk_element'].mean():.1%}")
print(f"Fraction of current Top 20 Ratio with magnetic risk: {top20_ratio['contains_magnetic_risk_element'].mean():.1%}")

def print_table(df_to_print, title):
    print("\n" + "="*80)
    print(title)
    print("="*80)
    for i, (_, row) in enumerate(df_to_print.head(20).iterrows(), 1):
        print(f"{i:2d}. {row['formula']:<15} | Tc = {row['tc']:5.2f} K | Unc = {row['unc']:5.2f} K | Ratio = {row['ratio']:5.2f} | ID = {row['material_id']}")
    if len(df_to_print) == 0:
        print(" (No candidates)")

# Split lists
df_no_mag = df_results[~df_results['contains_magnetic_risk_element']]
df_mag = df_results[df_results['contains_magnetic_risk_element']]

print_table(df_no_mag.sort_values(by='tc', ascending=False), "TOP 20 RANKED BY RAW PREDICTED Tc (NO MAGNETIC RISK ELEMENTS)")
print_table(df_mag.sort_values(by='tc', ascending=False), "TOP 20 RANKED BY RAW PREDICTED Tc (WITH MAGNETIC RISK ELEMENTS)")

print_table(df_no_mag[df_no_mag['tc'] > 1.0].sort_values(by='ratio', ascending=True), "TOP 20 RANKED BY LOWEST RATIO (Tc > 1K) (NO MAGNETIC RISK ELEMENTS)")
print_table(df_mag[df_mag['tc'] > 1.0].sort_values(by='ratio', ascending=True), "TOP 20 RANKED BY LOWEST RATIO (Tc > 1K) (WITH MAGNETIC RISK ELEMENTS)")
print("="*80 + "\n")

