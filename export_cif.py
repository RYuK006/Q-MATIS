import pandas as pd
import json
import os
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter

def export_candidate_cif(material_id: str, output_dir: str = "cif_exports") -> str:
    """
    Exports a candidate's structure from the MP reference dataset to a CIF file.
    Returns the path to the written CIF.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load from the MP reference dataset since our 43 candidates are from there
    df_mp = pd.read_parquet('mp_reference_hull_with_structs.parquet')
    row = df_mp[df_mp['material_id'] == material_id]
    
    if len(row) == 0:
        raise ValueError(f"Material ID {material_id} not found.")
        
    struct_dict_str = row.iloc[0]['structure_dict']
    struct = Structure.from_dict(json.loads(struct_dict_str))
    
    cif_path = os.path.join(output_dir, f"{material_id}.cif")
    
    # Write using pymatgen's CifWriter
    # Set symprec to a small value or use symprec=None to write P1 (exact structure)
    # Using symprec=0.01 is standard, but P1 is safest for ML models
    writer = CifWriter(struct, symprec=None)
    writer.write_file(cif_path)
    
    return cif_path

if __name__ == "__main__":
    # Test on one of the 2-structure test candidates: Sr2Co2O5 (mp-aaabfqpo)
    test_id = "mp-aaabfqpo"
    out_file = export_candidate_cif(test_id)
    print(f"Successfully exported {test_id} to {out_file}")
    print("\n--- CIF CONTENTS ---")
    with open(out_file, 'r') as f:
        print(f.read())
