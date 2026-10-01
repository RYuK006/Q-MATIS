from pymatgen.core import Composition, Structure, Lattice
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
import warnings
warnings.filterwarnings("ignore")

sp = SubstitutionPredictor(threshold=1e-8)

tests = [
    ({"Na+": 1, "Cl-": 1}, {"K+": 1, "Cl-": 1}, "Na -> K"),
    ({"Na+": 1, "Cl-": 1}, {"Na+": 1, "Br-": 1}, "Cl -> Br"),
    ({"Mg2+": 1, "O2-": 1}, {"Ca2+": 1, "O2-": 1}, "Mg -> Ca"),
    ({"Mg2+": 1, "O2-": 1}, {"Mg2+": 1, "S2-": 1}, "O -> S"),
    ({"Fe2+": 1, "O2-": 1}, {"Mn2+": 1, "O2-": 1}, "Fe2+ -> Mn2+")
]

print("Textbook Substitution Probabilities:")
probs = []
for start_dict, end_dict, desc in tests:
    comp1 = Composition(start_dict)
    
    # dummy structure to apply transformation
    coords = [[0,0,0]] * len(comp1.keys())
    st = Structure(Lattice.cubic(5), list(comp1.keys()), coords)
    
    preds = sp.composition_prediction(comp1, to_this_composition=False)
    found_prob = None
    for p in preds:
        try:
            t = SubstitutionTransformation(p['substitutions'])
            new_st = t.apply_transformation(st)
            new_st.remove_oxidation_states()
            if new_st.composition.reduced_formula == Composition(end_dict).reduced_formula:
                found_prob = p['probability']
                break
        except Exception:
            pass
            
    if found_prob:
        print(f"{desc:<15} : {found_prob:.6e}")
        probs.append(found_prob)
    else:
        print(f"{desc:<15} : Not found")

if probs:
    threshold = min(probs) * 0.99
    print(f"\nRecommended Threshold: {threshold:.6e}")
