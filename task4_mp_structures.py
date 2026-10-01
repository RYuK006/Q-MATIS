import os
import requests
import pandas as pd
import json

api_key = os.environ.get("MP_API_KEY", "h8yUPGpRyq8vEc8u5Ks5DH7KcA6YaX6S")
fields = "material_id,energy_above_hull,formation_energy_per_atom,uncorrected_energy_per_atom,nsites,formula_pretty,structure"
url = f"https://api.materialsproject.org/materials/summary/?_fields={fields}"
headers = {"X-API-KEY": api_key}

data = []
skip = 0
limit = 1000

print("Fetching MP summary with structures...")
while True:
    res = requests.get(f"{url}&_skip={skip}&_limit={limit}", headers=headers)
    if res.status_code != 200:
        print(f"Error {res.status_code}: {res.text}")
        break
    j = res.json()
    docs = j.get('data', [])
    if not docs:
        break
    
    for d in docs:
        if d['material_id'].startswith("mp-") or d['material_id'].startswith("mvc-"):
            data.append({
                "material_id": d["material_id"],
                "e_hull": d.get("energy_above_hull"),
                "e_form": d.get("formation_energy_per_atom"),
                "uncorrected_energy_per_atom": d.get("uncorrected_energy_per_atom"),
                "n_atoms": d.get("nsites"),
                "formula": d.get("formula_pretty"),
                "structure_dict": json.dumps(d["structure"]) if d.get("structure") else None
            })
    
    skip += limit
    print(f"Fetched {skip}, total MP filtered so far: {len(data)}", end="\r")

print("\nDone.")
df = pd.DataFrame(data)
df = df.dropna(subset=['e_hull', 'e_form', 'structure_dict', 'uncorrected_energy_per_atom'])
df.to_parquet("mp_reference_hull_with_structs.parquet")
print(f"Saved {len(df)} entries.")
