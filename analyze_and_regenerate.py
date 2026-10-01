import pandas as pd
import json
from pymatgen.core import Structure
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
import warnings
warnings.filterwarnings("ignore")

# Load previous 421 candidates
df_old = pd.read_parquet("novel_candidates_results.parquet")
print("Original Candidate Distribution (421 candidates, threshold=1e-3):")
dist = df_old['prototype_id'].value_counts()
for p, count in dist.items():
    print(f"  {p}: {count}")

top10_formulas = set(df_old.sort_values('e_hull_pred_mace')['formula'].head(10).tolist())

df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

# 1. Select 10 prototypes exactly like run_pipeline.py does
prototypes = []
for _, row in df_mp.head(100).iterrows():
    if row['n_atoms'] > 20:
        continue
    st = Structure.from_dict(json.loads(row['structure_dict']))
    try:
        st.add_oxidation_state_by_guess()
        prototypes.append((row['material_id'], row['formula'], st))
        if len(prototypes) == 10:
            break
    except Exception:
        pass

# 2. Generate with higher threshold
sp = SubstitutionPredictor(threshold=0.05)
candidates = []
matcher = StructureMatcher()

print("\nGenerating new candidates with threshold=0.05...")
for pid, pform, pst in prototypes:
    if len(candidates) >= 500:
        break
        
    try:
        preds = sp.composition_prediction(pst.composition, to_this_composition=False)
    except Exception:
        continue
    for p in preds:
        if len(candidates) >= 500:
            break
        try:
            t = SubstitutionTransformation(p['substitutions'])
            new_st = t.apply_transformation(pst)
            new_st.remove_oxidation_states()
            new_formula = new_st.composition.reduced_formula
            
            # Deduplicate against MP
            mp_matches = df_mp[df_mp['formula'] == new_formula]
            is_known = False
            for _, m_row in mp_matches.iterrows():
                m_st = Structure.from_dict(json.loads(m_row['structure_dict']))
                if matcher.fit(new_st, m_st):
                    is_known = True
                    break
            
            if not is_known:
                # Also deduplicate internally by formula for a cleaner list
                if not any(c['formula'] == new_formula for c in candidates):
                    candidates.append({
                        'formula': new_formula,
                        'prototype_id': pid,
                        'probability': p['probability']
                    })
        except Exception:
            pass

df_cand = pd.DataFrame(candidates)
print(f"\nNew candidate count: {len(candidates)}")
if not df_cand.empty:
    dist = df_cand['prototype_id'].value_counts()
    print("New Prototype Distribution:")
    for pid, count in dist.items():
        print(f"  {pid}: {count}")

    print("\nCandidates:")
    for _, row in df_cand.iterrows():
        print(f"  {row['formula']:<15} (from {row['prototype_id']}) - Prob: {row['probability']:.6f}")
        
    survivors = set(df_cand['formula']).intersection(top10_formulas)
    if survivors:
        print("\nPrevious Top 10 Survivors:")
        for s in survivors:
            print(f"  {s}")
    else:
        print("\nNone of the previous Top 10 candidates survived the 5% threshold.")
