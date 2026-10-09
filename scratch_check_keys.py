import json
import pandas as pd

df_db = pd.read_json('BETE-NET-zip/BETE-NET-main/database.json')

with open('BETE-NET-zip/BETE-NET-main/test_preds/CPD.json') as f:
    cpd = json.load(f)

with open('BETE-NET-zip/BETE-NET-main/test_preds/CSO.json') as f:
    cso = json.load(f)

print("Checking sample test IDs across db, CPD, CSO:")
for test_id in ['71', '0', '148', '297', '175']:
    tid_int = int(test_id)
    comp_db = df_db.loc[tid_int, 'comp'] if tid_int in df_db.index else 'NOT_IN_DB'
    comp_cpd = cpd['comp'].get(test_id, 'NOT_IN_CPD')
    comp_cso = cso['comp'].get(test_id, 'NOT_IN_CSO')
    print(f"ID {test_id}: db={comp_db}, cpd={comp_cpd}, cso={comp_cso}")

# What are the missing test IDs 36, 786, 218 in CSO and CPD?
for mid in ['36', '786', '218']:
    print(f"Missing ID {mid}: in db? {int(mid) in df_db.index}, in cpd? {mid in cpd['comp']}, in cso? {mid in cso['comp']}")
