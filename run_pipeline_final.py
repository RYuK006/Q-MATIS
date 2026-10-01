import json
import time
import pandas as pd
from pymatgen.core import Structure, Composition
from pymatgen.analysis.structure_prediction.substitution_probability import SubstitutionPredictor
from pymatgen.transformations.standard_transformations import SubstitutionTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from mace.calculators import mace_mp
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
import torch
import json

EXCLUSION_LIST = set([
    "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr",
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra",
    "Tl", "Hg", "Cd", "Os", "Ir"
])

def run_pipeline():
    print("Loading MP reference hull...")
    df_mp = pd.read_parquet("mp_reference_hull_with_structs.parquet")
    
    # 1. Build Phase Diagram using uncorrected energies
    # This allows us to perfectly evaluate MACE-MP's uncorrected raw output
    print("Parsing MP reference entries...")
    raw_entries = []
    terminal_elements = set()
    for _, row in df_mp.iterrows():
        try:
            comp = Composition(row['formula'])
            energy = row['uncorrected_energy_per_atom'] * comp.num_atoms
            raw_entries.append(PDEntry(comp, energy))
            if len(comp.elements) == 1:
                terminal_elements.add(list(comp.elements)[0].symbol)
        except Exception:
            pass
            
    pd_entries = [e for e in raw_entries if all(el.symbol in terminal_elements for el in e.composition.elements)]
    
    # 2. Find prototypes
    # We select oxides/ceramics from the actually stable (e_hull=0) structures
    stable_df = df_mp[df_mp['e_hull'] <= 1e-4]
    
    def get_num_elements(form):
        try:
            return len(Composition(form).elements)
        except:
            return 0
            
    print("Sorting prototypes by number of distinct elements...")
    df_mp['n_elements'] = df_mp['formula'].apply(get_num_elements)
    df_mp_prototypes = df_mp[df_mp['n_atoms'] <= 20].sort_values(by='n_elements', ascending=False)
    
    prototypes = []
    print("Preparing up to 10,000 diverse prototypes...")
    for _, row in df_mp_prototypes.iterrows():
        try:
            st = Structure.from_dict(json.loads(row['structure_dict']))
            st.add_oxidation_state_by_guess()
            prototypes.append((row['material_id'], row['formula'], st))
            if len(prototypes) == 10000:
                break
        except Exception:
            pass
            
    print(f"Found {len(prototypes)} prototypes.")
    
    # 3. Generate Candidates
    sp = SubstitutionPredictor(threshold=0.0138)
    candidates = []
    matcher = StructureMatcher()
    seen_formulas = set()
    
    print("Generating candidates and deduplicating...")
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
                
                if any(el.symbol in EXCLUSION_LIST for el in new_st.composition.elements):
                    continue
                
                # Deduplicate internally first
                if new_formula in seen_formulas:
                    continue
                    
                # Deduplicate against MP using StructureMatcher
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
                        "prototype_id": pid,
                        "formula": new_formula,
                        "structure": new_st
                    })
            except Exception:
                pass

    print(f"Total novel candidates to relax: {len(candidates)}")
    
    # 4. Relax with MACE-MP
    print("Loading MACE-MP...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    
    macemp = mace_mp(model="medium", dispersion=False, default_dtype="float32", device=device)
    
    results = []
    import os
    checkpoint_file = "mace_checkpoint.csv"
    
    # Check if checkpoint exists and load previously relaxed formulas
    completed_formulas = set()
    if os.path.exists(checkpoint_file):
        try:
            df_ckpt = pd.read_csv(checkpoint_file)
            completed_formulas = set(df_ckpt['formula'].tolist())
            results = df_ckpt.to_dict('records')
            print(f"Loaded {len(completed_formulas)} completed candidates from checkpoint.")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    else:
        # Create checkpoint file with headers
        with open(checkpoint_file, 'w') as f:
            f.write("prototype_id,formula,raw_energy_per_atom,e_hull_pred_mace,category\n")
    
    from pymatgen.io.ase import AseAtomsAdaptor
    print("Starting relaxation...")
    for i, cand in enumerate(candidates):
        if cand['formula'] in completed_formulas:
            print(f"[{i+1}/{len(candidates)}] Skipping {cand['formula']} (already relaxed).")
            continue
            
        try:
            atoms = AseAtomsAdaptor.get_atoms(cand['structure'])
            atoms.calc = macemp
            cell_filter = FrechetCellFilter(atoms)
            opt = FIRE(cell_filter, logfile=None)
            opt.run(fmax=0.05, steps=500)
            
            e_total = atoms.get_potential_energy()
            n_atoms = len(atoms)
            comp = cand['structure'].composition
            
            # Build a localized PhaseDiagram for this chemical system to avoid OOM
            entry = PDEntry(comp, e_total)
            system_elements = set(el.symbol for el in comp.elements)
            local_entries = [e for e in pd_entries if set(el.symbol for el in e.composition.elements).issubset(system_elements)]
            local_pd = PhaseDiagram(local_entries)
            e_hull = local_pd.get_e_above_hull(entry)
            
            # Determine stability category
            if e_hull <= 0:
                cat = 'stable'
            elif e_hull <= 0.025:
                cat = 'metastable'
            else:
                cat = 'unstable'
                
            relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
            res_dict = {
                'formula': cand['formula'],
                'prototype_id': cand['prototype_id'],
                'raw_energy_per_atom': e_total / n_atoms,
                'e_hull_pred_mace': e_hull,
                'category': cat,
                'relaxed_structure': json.dumps(relaxed_struct.as_dict())
            }
            results.append(res_dict)
            
            # Save to checkpoint
            with open(checkpoint_file, 'a') as f:
                f.write(f"{res_dict['prototype_id']},{res_dict['formula']},{res_dict['raw_energy_per_atom']},{res_dict['e_hull_pred_mace']},{res_dict['category']}\n")
                
            print(f"[{i+1}/{len(candidates)}] {cand['formula']}: E_hull = {e_hull:.3f} eV/atom ({cat})")
        except Exception as e:
            print(f"Relaxation failed for {cand['formula']}: {e}")

    df_res = pd.DataFrame(results)
    if not df_res.empty:
        df_res = df_res.sort_values('e_hull_pred_mace')
        df_res.to_parquet("novel_candidates_results.parquet")
        print(f"\nSaved {len(df_res)} relaxed candidates to novel_candidates_results.parquet")
        
        print("\nTop 10 most stable novel candidates:")
        print(df_res.head(10).to_string(index=False))
    else:
        print("\nNo novel candidates were successfully relaxed.")

if __name__ == "__main__":
    run_pipeline()
