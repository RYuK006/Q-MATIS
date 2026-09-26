import json
import sqlite3
import pandas as pd
from pymatgen.core import Composition
import time

def main():
    print("Loading data...")
    # 1. Total SuperCon items
    df = pd.read_csv("data/supercon.csv")
    total = len(df)
    
    # 2. PyMatGen Parse Failures
    parse_fails = 0
    valid_formulas = set()
    for f in df['name'].fillna(""):
        try:
            c = Composition(f)
            if len(c.elements) > 0:
                valid_formulas.add(c.reduced_formula)
            else:
                parse_fails += 1
        except Exception:
            parse_fails += 1

    # 3. MP API Cache Analysis
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
            
    reasons = {
        "Unparseable Formula (SuperCon notation error)": 0,
        "No stable polymorph in Materials Project": 0,
        "Successfully Resolved": 0
    }
    
    for f in df['name'].fillna(""):
        try:
            c = Composition(f)
            if len(c.elements) > 0:
                red_f = c.reduced_formula
                if red_f in mp_formulas:
                    reasons["Successfully Resolved"] += 1
                else:
                    reasons["No stable polymorph in Materials Project"] += 1
            else:
                reasons["Unparseable Formula (SuperCon notation error)"] += 1
        except Exception:
            reasons["Unparseable Formula (SuperCon notation error)"] += 1

    print(f"Total: {total}")
    for k, v in reasons.items():
        print(f"{k}: {v}")
        
if __name__ == "__main__":
    main()
