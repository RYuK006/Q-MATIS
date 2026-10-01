with open('run_pipeline.py', 'r') as f:
    code = f.read()

old_proto_logic = '''    def is_ceramic(formula):
        comp = Composition(formula)
        return any(el.symbol in ['O', 'S', 'Se', 'N', 'P'] for el in comp.elements)
    
    ceramic_df = stable_df[stable_df['formula'].apply(is_ceramic)]
    
    prototypes = []
    print("Finding 10 oxide/ceramic prototypes...")
    for _, row in ceramic_df.iterrows():
        st = Structure.from_dict(json.loads(row['structure_dict']))
        try:
            st.add_oxidation_state_by_guess()
            prototypes.append((row['material_id'], row['formula'], st))
            if len(prototypes) == 10:
                break
        except Exception as e:
            print(f"Skipping prototype {row['material_id']} ({row['formula']}) due to oxidation guess failure: {e}")'''

new_proto_logic = '''    def get_num_elements(form):
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
            pass'''
code = code.replace(old_proto_logic, new_proto_logic)

# Replace generation params
code = code.replace('sp = SubstitutionPredictor(threshold=1e-3)', 'sp = SubstitutionPredictor(threshold=0.0138)')

# Remove the 500 cap
code = code.replace('''            if len(candidates) >= 500:
                break
''', '')

with open('run_pipeline_final.py', 'w') as f:
    f.write(code)
print('Patched successfully to run_pipeline_final.py')
