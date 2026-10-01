import pandas as pd
import json
import numpy as np
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
import time
import warnings
warnings.filterwarnings("ignore")

df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

def get_num_elements(form):
    try:
        return len(Composition(form).elements)
    except:
        return 0

print("Sorting prototypes by number of distinct elements...")
df_mp['n_elements'] = df_mp['formula'].apply(get_num_elements)
df_mp = df_mp[df_mp['n_atoms'] <= 20] 
df_mp = df_mp.sort_values(by='n_elements', ascending=False)

prototypes = []
for _, row in df_mp.iterrows():
    try:
        st = Structure.from_dict(json.loads(row['structure_dict']))
        st.add_oxidation_state_by_guess()
        prototypes.append((row['material_id'], row['formula'], st))
        if len(prototypes) == 10000:
            break
    except Exception:
        continue

print(f"Selected {len(prototypes)} prototypes.")

sp = SubstitutionPredictor(threshold=0.0138)
candidates = []
matcher = StructureMatcher()
seen_formulas = set()

print("Generating candidates with threshold=0.0138...")
start_time = time.time()
for pid, pform, pst in prototypes:
    try:
        preds = sp.composition_prediction(pst.composition, to_this_composition=False)
    except Exception:
        continue
        
    for p in preds:
        try:
            t = SubstitutionTransformation(p['substitutions'])
            new_st = t.apply_transformation(pst)
            new_st.remove_oxidation_states()
            new_formula = new_st.composition.reduced_formula
            
            if new_formula in seen_formulas:
                continue
                
            mp_matches = df_mp[df_mp['formula'] == new_formula]
            is_known = False
            for _, m_row in mp_matches.iterrows():
                m_st = Structure.from_dict(json.loads(m_row['structure_dict']))
                if matcher.fit(new_st, m_st):
                    is_known = True
                    break
            
            if not is_known:
                seen_formulas.add(new_formula)
                candidates.append({
                    'formula': new_formula,
                    'prototype_id': pid,
                    'probability': p['probability']
                })
        except Exception:
            pass

end_time = time.time()
print(f"\nSubstitutionPredictor scan took {end_time - start_time:.2f} seconds.")

df_cand = pd.DataFrame(candidates)
print(f"\nTotal candidates generated at 0.0138 threshold: {len(candidates)}")

if not df_cand.empty:
    print("\nPrototype Diversity:")
    dist = df_cand['prototype_id'].value_counts()
    for pid, count in dist.items():
        print(f"  {pid}: {count}")

    print("\nProbability Score Distribution:")
    print(f"  Min:    {df_cand['probability'].min():.6f}")
    print(f"  Median: {df_cand['probability'].median():.6f}")
    print(f"  Max:    {df_cand['probability'].max():.6f}")
