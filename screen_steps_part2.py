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

print("=== Loading Previous Data ===")
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
stable_mp = df_mp[df_mp['e_hull'] <= 1e-5]

sc_df = pd.read_csv("data/supercon.csv")
def get_element_set(formula):
    try:
        return frozenset([el.symbol for el in Composition(formula).elements])
    except:
        return frozenset()
        
sc_element_sets = set(sc_df['name'].dropna().apply(get_element_set))
stable_mp['element_set'] = stable_mp['formula'].apply(get_element_set)
dedup_mp = stable_mp[~stable_mp['element_set'].isin(sc_element_sets)]

material_ids = dedup_mp['material_id'].tolist()
print(f"Deduplicated pool size: {len(material_ids)}")

print("\n=== STEP 1: Fetch Band Gaps ===")
# Fetch band gaps
try:
    with MPRester(mp_api_key) as mpr:
        print("Fetching all stable materials from MP API to match by formula...")
        docs = mpr.materials.summary.search(
            energy_above_hull=(-1e-5, 1e-5),
            fields=["formula_pretty", "band_gap"]
        )
        
    bg_data = {str(d.formula_pretty): getattr(d, 'band_gap', None) for d in docs}
    
    dedup_mp['band_gap'] = dedup_mp['formula'].apply(lambda x: bg_data.get(str(x), None))
    
    # Drop rows where we couldn't fetch band_gap (should be 0)
    has_bg = dedup_mp.dropna(subset=['band_gap'])
    
    bg_zero = (has_bg['band_gap'] == 0).sum()
    bg_gt_zero = (has_bg['band_gap'] > 0).sum()
    
    print(f"Entries with band_gap data: {len(has_bg)}")
    print(f"Band gap == 0: {bg_zero}")
    print(f"Band gap > 0:  {bg_gt_zero}")
    
except Exception as e:
    print(f"Failed to fetch band gaps: {e}")
    # In case of failure, exit early
    import sys
    sys.exit(1)

print("\n=== STEP 2: Filter Metallic Candidates ===")
metallic_mp = has_bg[has_bg['band_gap'] <= 0.1]
print(f"Pool size after band_gap <= 0.1 eV filter: {len(metallic_mp)}")

print("\n=== STEP 3: Sanity Test (Metallic Only) ===")
# Define featurizer
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

try:
    sample_100 = metallic_mp.sample(100, random_state=42).copy()
    
    model = joblib.load("models/rf_tc_model.joblib")
    
    tcs = []
    stds = []
    failed = 0
    
    for formula in sample_100['formula']:
        feat = get_composition_features(formula)
        if feat is None:
            failed += 1
            continue
            
        feat_reshaped = feat.reshape(1, -1)
        pred_tc = model.predict(feat_reshaped)[0]
        
        if hasattr(model, "estimators_"):
            tree_preds = [t.predict(feat_reshaped)[0] for t in model.estimators_]
            pred_std = np.std(tree_preds)
        else:
            pred_std = 0.0
            
        tcs.append(pred_tc)
        stds.append(pred_std)
        
    print(f"Parsing failures: {failed}")
    if tcs:
        print(f"Predicted Tc distribution (Metallic):")
        print(f"  Min: {np.min(tcs):.2f} K")
        print(f"  Median: {np.median(tcs):.2f} K")
        print(f"  Max: {np.max(tcs):.2f} K")
        print(f"Predicted Uncertainty distribution (Metallic):")
        print(f"  Min: {np.min(stds):.2f} K")
        print(f"  Median: {np.median(stds):.2f} K")
        print(f"  Max: {np.max(stds):.2f} K")
        
        # Compare ratios loosely
        print(f"\nAnalysis: For the metallic sample, median Tc is {np.median(tcs):.2f} and median Unc is {np.median(stds):.2f}.")
except Exception as e:
    print(f"Step 3 failed: {e}")
