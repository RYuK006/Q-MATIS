from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.core import Structure
import pandas as pd
import json

df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")

def get_proto_struct(mpid):
    row = df_mp[df_mp['material_id'] == mpid].iloc[0]
    st = Structure.from_dict(json.loads(row['structure_dict']))
    st.add_oxidation_state_by_guess()
    return st

sp = SubstitutionPredictor(threshold=1e-3)

targets = ['BHNF', 'MgH4(OF)2', 'TiH4(OF)2', 'CuH4(OF)2', 'ScH4(OF)2']
prototypes = {
    'BHNF': 'mp-aaahikwu',
    'MgH4(OF)2': 'mp-aaahikra',
    'TiH4(OF)2': 'mp-aaahikra',
    'CuH4(OF)2': 'mp-aaahikra',
    'ScH4(OF)2': 'mp-aaahikra'
}

print("Calculating substitution probabilities...")
for cand_form in targets:
    mpid = prototypes[cand_form]
    try:
        pst = get_proto_struct(mpid)
        preds = sp.composition_prediction(pst.composition, to_this_composition=False)
        found = False
        for p in preds:
            t = SubstitutionTransformation(p['substitutions'])
            new_st = t.apply_transformation(pst)
            new_st.remove_oxidation_states()
            if new_st.composition.reduced_formula == cand_form:
                print(f"{cand_form:>12} (from {pst.composition.reduced_formula:<12}): Probability = {p['probability']:.6e}")
                found = True
                break
        if not found:
            print(f"{cand_form:>12} (from {pst.composition.reduced_formula:<12}): Not found (prob below 1e-6)")
    except Exception as e:
        print(f"{cand_form}: Error - {e}")
