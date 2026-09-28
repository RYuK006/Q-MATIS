import os
import requests
import pandas as pd
from dotenv import load_dotenv
import time

load_dotenv()
api_key = os.environ.get("MP_API_KEY")

url = "https://api.materialsproject.org/materials/summary/"
headers = {"X-API-KEY": api_key, "accept": "application/json"}
params = {
    "_fields": "material_id,energy_above_hull,nsites,formula_pretty,origins",
    "_limit": 1000,
    "_skip": 0
}

all_data = []
print("Fetching MP summary...")
while True:
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        docs = data.get("data", [])
        if not docs:
            break
        
        for doc in docs:
            is_alex = False
            origins = doc.get("origins") or []
            for org in origins:
                name = org.get("name", "") if isinstance(org, dict) else str(org)
                if "alexandria" in name.lower():
                    is_alex = True
            if is_alex:
                continue
                
            all_data.append({
                "material_id": doc.get("material_id"),
                "e_hull": doc.get("energy_above_hull"),
                "n_atoms": doc.get("nsites"),
                "formula": doc.get("formula_pretty")
            })
            
        print(f"Fetched {params['_skip'] + len(docs)}, total filtered so far: {len(all_data)}", end='\r')
        
        if len(docs) < params["_limit"]:
            break
        params["_skip"] += params["_limit"]
        
    except Exception as e:
        print(f"\nError: {e}, retrying in 5s...")
        time.sleep(5)
print("\nDone.")

df = pd.DataFrame(all_data)
df.to_parquet("mp_reference_hull.parquet")

print(f"Total entries: {len(df)}")
stable_count = (df['e_hull'] <= 0.0).sum()
print(f"Entries with E_hull == 0: {stable_count}")

bins = [0, 4, 10, 20, 40, float('inf')]
labels = ["<=4", "5-10", "11-20", "21-40", ">40"]
df['atom_bucket'] = pd.cut(df['n_atoms'], bins=bins, labels=labels)
print("Atom-count distribution:")
print(df['atom_bucket'].value_counts().sort_index())
