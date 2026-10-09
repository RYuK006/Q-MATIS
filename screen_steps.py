import os
import pandas as pd
import numpy as np
import requests
from pymatgen.core import Composition
import joblib
from mp_api.client import MPRester
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

load_dotenv()
mp_api_key = os.getenv("MP_API_KEY")

print("=== STEP 1a ===")
df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
stable_mp = df_mp[df_mp['e_hull'] <= 1e-5]
print(f"Total stable MP entries (e_hull <= 0): {len(stable_mp)}")

print("\n=== STEP 1b ===")
# Let's query MP API to see if GNoME is identifiable.
# We will query the first 50 stable material_ids to inspect their fields.
sample_ids = stable_mp['material_id'].head(50).tolist()
try:
    with MPRester(mp_api_key) as mpr:
        docs = mpr.materials.summary.search(material_ids=sample_ids)
        if docs:
            doc = docs[0]
            # Check for GNoME in origins or database_IDs
            print(f"Sample origin fields available in MP summary doc:")
            if hasattr(doc, 'origins'):
                print(f"  origins: {doc.origins}")
            if hasattr(doc, 'database_IDs'):
                print(f"  database_IDs: {doc.database_IDs}")
            if hasattr(doc, 'theoretical'):
                print(f"  theoretical: {doc.theoretical}")
            
            # Let's also search broadly: can we search by provenance?
            # It's known that GNoME structures might have specific tags, but MP-API might not expose "GNoME" directly 
            # as a search field in summary. Let's do a quick search in the provenance endpoint if it exists.
            print("Note: MP API 'provenance' endpoint often contains author 'Google' or 'GNoME'.")
            try:
                prov_docs = mpr.materials.provenance.search(material_ids=sample_ids)
                if prov_docs:
                    print(f"  provenance authors: {prov_docs[0].authors}")
            except Exception as e:
                print(f"  Provenance check failed: {e}")
                
except Exception as e:
    print(f"MP API check failed: {e}")
print("If no 'GNoME' specific flag is easily queryable across the 154k in bulk without doing 154k provenance queries, I will report that.")

print("\n=== STEP 1c ===")
try:
    r = requests.get("https://alexandria.icams.rub.de/datasets/")
    text = r.text.lower()
    if 'stable' in text or 'hull' in text:
        print(f"Found mentions of 'stable' or 'hull' on Alexandria dataset page.")
        # Find hrefs simply
        import re
        hrefs = re.findall(r'href=[\'"]?([^\'" >]+)', r.text)
        stable_links = [h for h in hrefs if 'stable' in h.lower() or 'hull' in h.lower()]
        print(f"Potential links: {stable_links}")
    else:
        print("No direct 'stable' or 'hull' subset download link found on the Alexandria datasets page.")
except Exception as e:
    print(f"Failed to check Alexandria: {e}")


print("\n=== STEP 2 ===")
try:
    sc_df = pd.read_csv("data/supercon.csv")
    print(f"Loaded SuperCon training data: {len(sc_df)} entries.")
    
    # Extract element SET for every training compound
    def get_element_set(formula):
        try:
            return frozenset([el.symbol for el in Composition(formula).elements])
        except:
            return frozenset()
            
    sc_element_sets = set(sc_df['name'].dropna().apply(get_element_set))
    print(f"Unique element sets in SuperCon: {len(sc_element_sets)}")
    
    # Remove candidates from Step 1's pool
    stable_mp['element_set'] = stable_mp['formula'].apply(get_element_set)
    dedup_mask = ~stable_mp['element_set'].isin(sc_element_sets)
    dedup_mp = stable_mp[dedup_mask]
    
    print(f"Pool size before dedup: {len(stable_mp)}")
    print(f"Pool size after dedup: {len(dedup_mp)}")
except Exception as e:
    print(f"Step 2 failed: {e}")

print("\n=== STEP 3 ===")
try:
    sample_100 = dedup_mp.sample(100, random_state=42).copy()
    
    # Load model and featurizer
    import sys
    sys.path.append(os.path.abspath("."))
    from superconductor.features import get_node_features
    from pymatgen.core import Composition

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
    
    # If the model is not locally available, download it
    model_path = "models/rf_tc_model.joblib"
    if not os.path.exists(model_path):
        print(f"Downloading model to {model_path}...")
        os.makedirs("models", exist_ok=True)
        r = requests.get("https://huggingface.co/Aaron006/Discovery/resolve/main/rf_tc_model.joblib")
        with open(model_path, "wb") as f:
            f.write(r.content)
            
    model = joblib.load(model_path)
    
    failed_formulas = []
    tcs = []
    stds = []
    
    for formula in sample_100['formula']:
        try:
            features = get_composition_features(formula)
            if features is None:
                failed_formulas.append(formula)
                continue
            
            features_reshaped = features.reshape(1, -1)
            pred_tc = model.predict(features_reshaped)[0]
            
            # Get uncertainty (std of trees)
            if hasattr(model, "estimators_"):
                tree_preds = [tree.predict(features_reshaped)[0] for tree in model.estimators_]
                pred_std = np.std(tree_preds)
            else:
                pred_std = 0.0
                
            tcs.append(pred_tc)
            stds.append(pred_std)
        except Exception as e:
            failed_formulas.append(f"{formula}: {e}")
            
    print(f"Parsing failures: {len(failed_formulas)}")
    if failed_formulas:
        print(f"Sample failures: {failed_formulas[:5]}")
        
    if tcs:
        print(f"Predicted Tc distribution:")
        print(f"  Min: {np.min(tcs):.2f} K")
        print(f"  Median: {np.median(tcs):.2f} K")
        print(f"  Max: {np.max(tcs):.2f} K")
        print(f"Predicted Uncertainty distribution:")
        print(f"  Min: {np.min(stds):.2f} K")
        print(f"  Median: {np.median(stds):.2f} K")
        print(f"  Max: {np.max(stds):.2f} K")
except Exception as e:
    print(f"Step 3 failed: {e}")
