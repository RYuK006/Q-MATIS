from download_model import download_if_missing
import itertools
import numpy as np
import time
from pymatgen.core import Composition, Element
import multiprocessing
import gc
import joblib

import sys
sys.path.append('.')
from run_comp_model import get_composition_features

def get_valid_anions():
    return ['O', 'S', 'Se', 'Te', 'N', 'P', 'As', 'Sb', 'F', 'Cl', 'Br', 'I']

def get_valid_metals():
    # Roughly 55-60 metals, avoiding actinides and highly radioactive
    excluded = {'U', 'Pu', 'Th', 'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Pa', 'Np', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Tl', 'Hg', 'Cd', 'Os', 'Ir'}
    metals = []
    for el in Element:
        if el.is_metal and el.symbol not in excluded:
            metals.append(el.symbol)
    return metals

def get_stoichiometries(max_sum=10):
    stoichs = []
    for a in range(1, max_sum):
        for b in range(1, max_sum):
            for c in range(1, max_sum):
                for d in range(1, max_sum):
                    if a + b + c + d <= max_sum:
                        # only keep reduced stoichiometries to avoid duplicates (e.g. A2B2C2D2 is just ABCD scaled)
                        if np.gcd.reduce([a,b,c,d]) == 1:
                            stoichs.append((a, b, c, d))
    return stoichs

def check_charge_balance(combo, stoich):
    formula_str = f"{combo[0]}{stoich[0]}{combo[1]}{stoich[1]}{combo[2]}{stoich[2]}{combo[3]}{stoich[3]}"
    try:
        comp = Composition(formula_str)
        # oxi_state_guesses returns a list of valid oxidation state assignments
        guesses = comp.oxi_state_guesses(max_sites=-1) # fast
        return len(guesses) > 0
    except Exception:
        return False

def count_and_sample_combinations():
    metals = get_valid_metals()
    anions = get_valid_anions()
    stoichs = get_stoichiometries(10)
    
    metal_combos = list(itertools.combinations(metals, 3))
    
    total_raw = len(metal_combos) * len(anions) * len(stoichs)
    print(f"Total raw combinations (3 metals + 1 anion, max_atoms=10): {total_raw}")
    
    # We will sample 1,000,000 random raw combinations to estimate the charge balance pass rate
    import random
    random.seed(42)
    sample_size = 100000
    
    passes = 0
    
    # Sample combinations
    print(f"Sampling {sample_size} candidates for charge balance check...")
    t0 = time.time()
    
    passed_samples = []
    
    for _ in range(sample_size):
        m_combo = random.choice(metal_combos)
        anion = random.choice(anions)
        combo = (m_combo[0], m_combo[1], m_combo[2], anion)
        stoich = random.choice(stoichs)
        
        if check_charge_balance(combo, stoich):
            passes += 1
            formula_str = f"{combo[0]}{stoich[0]}{combo[1]}{stoich[1]}{combo[2]}{stoich[2]}{combo[3]}{stoich[3]}"
            passed_samples.append(formula_str)
            
    pass_rate = passes / sample_size
    projected_valid = int(total_raw * pass_rate)
    print(f"Charge balance pass rate: {pass_rate:.4f} ({passes} out of {sample_size})")
    print(f"Projected viable Phase 1 candidates: {projected_valid:,}")
    print(f"Time taken for sample check: {time.time()-t0:.2f}s")
    
    return passed_samples, total_raw, projected_valid

def main():
    passed_samples, total_raw, projected_valid = count_and_sample_combinations()
    
    print("\nFeaturizing and scoring the passed sample (up to 10k)...")
    
    model = joblib.load(download_if_missing())
    
    to_score = passed_samples[:10000]
    
    X = []
    valid = []
    for f in to_score:
        feats = get_composition_features(f)
        if feats is not None:
            X.append(feats)
            valid.append(f)
            
    X = np.array(X)
    preds = model.predict(X)
    
    from test_ood import get_rf_uncertainty
    uncerts = get_rf_uncertainty(model, X)
    
    results = []
    for i, f in enumerate(valid):
        results.append((f, preds[i], uncerts[i]))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Top 20 General Phase 1 Candidates (from Sample) ---")
    for i in range(20):
        f, tc, u = results[i]
        print(f"{i+1:2}. {f:20} -> Predicted Tc = {tc:6.2f} ± {u:5.2f} K")
        
if __name__ == "__main__":
    main()
