import json
import sqlite3
import pandas as pd
from pymatgen.core import Composition

def main():
    df = pd.read_csv("data/supercon.csv")
    
    conn = sqlite3.connect("cache/dataset/materials_cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT formula FROM structures WHERE source='MP'")
    mp_rows = cursor.fetchall()
    
    mp_formulas = set()
    for row in mp_rows:
        try:
            mp_formulas.add(Composition(row[0]).reduced_formula)
        except:
            pass
            
    rejected = []
    
    for f in df['name'].fillna(""):
        try:
            c = Composition(f)
            if len(c.elements) > 0:
                red_f = c.reduced_formula
                if red_f not in mp_formulas:
                    rejected.append((f, red_f))
                    if len(rejected) >= 10:
                        break
        except Exception:
            pass

    print("10 Rejected Examples:")
    for i, (orig, red) in enumerate(rejected):
        print(f"{i+1}. Original: {orig}, Reduced: {red}")
        
if __name__ == "__main__":
    main()
