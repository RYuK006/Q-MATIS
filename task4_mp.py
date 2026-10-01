import os
import pandas as pd
from mp_api.client import MPRester
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ.get("MP_API_KEY")

print("Querying Materials Project...")
with MPRester(api_key) as mpr:
    # Use MPRester.materials.summary.search
    docs = mpr.materials.summary.search(
        fields=["material_id", "energy_above_hull", "nsites", "formula_pretty", "origins", "is_stable"]
    )

data = []
for doc in docs:
    # Filter out Alexandria if it's in origins
    if doc.origins:
        # origins is a list of Origin objects or dicts
        # just check if any origin name is "alexandria"
        is_alex = False
        for org in doc.origins:
            if hasattr(org, "name") and "alexandria" in str(org.name).lower():
                is_alex = True
            elif isinstance(org, dict) and "alexandria" in str(org.get("name", "")).lower():
                is_alex = True
            elif "alexandria" in str(org).lower():
                is_alex = True
        if is_alex:
            continue
            
    data.append({
        "material_id": str(doc.material_id),
        "e_hull": doc.energy_above_hull,
        "n_atoms": doc.nsites,
        "formula": doc.formula_pretty
    })

df = pd.DataFrame(data)
df.to_parquet("mp_reference_hull.parquet")

print(f"Total entries: {len(df)}")
stable_count = (df['e_hull'] <= 0.0).sum()
print(f"Entries with E_hull == 0: {stable_count}")

bins = [0, 4, 10, 20, 40, float('inf')]
labels = ["<=4", "5-10", "11-20", "21-40", ">40"]
df['atom_bucket'] = pd.cut(df['n_atoms'], bins=bins, labels=labels)
print("Atom-count distribution:")
print(df['atom_bucket'].value_counts().sort_index())
