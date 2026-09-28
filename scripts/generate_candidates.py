import os
import csv
import itertools
from pymatgen.core import Composition
import numpy as np

def generate_combinatorial_space():
    """
    Generates physically reasonable candidate compositions by varying
    stoichiometry in known superconducting families and simple ternary/quaternary spaces.
    """
    candidates = set()
    
    # 1. Cuprate-like space (Y-Ba-Cu-O variations)
    # Y(1-x) Nd(x) Ba2 Cu3 O(7-d)
    rare_earths = ['Y', 'Nd', 'La', 'Sm', 'Gd']
    for re in rare_earths:
        for x in np.arange(0.0, 1.1, 0.1):
            for d in np.arange(0.0, 0.5, 0.1):
                # We format it by combining amounts
                y_amt = round(1.0 - x, 2)
                nd_amt = round(x, 2)
                o_amt = round(7.0 - d, 2)
                
                parts = []
                if y_amt > 0: parts.append(f"Y{y_amt}")
                if nd_amt > 0: parts.append(f"{re}{nd_amt}")
                parts.append("Ba2")
                parts.append("Cu3")
                parts.append(f"O{o_amt}")
                candidates.add("".join(parts))

    # 2. Iron-pnictide space (Ba-K-Fe-As variations)
    # Ba(1-x) K(x) Fe2 As2
    dopants = ['K', 'Na', 'Rb', 'Cs']
    for dopant in dopants:
        for x in np.arange(0.0, 1.1, 0.05):
            ba_amt = round(1.0 - x, 2)
            d_amt = round(x, 2)
            parts = []
            if ba_amt > 0: parts.append(f"Ba{ba_amt}")
            if d_amt > 0: parts.append(f"{dopant}{d_amt}")
            parts.append("Fe2As2")
            candidates.add("".join(parts))

    # 3. Simple binary / ternary exploration
    # Just picking some known good elements and sweeping fractions
    # (Nb, Ti, Zr, V, Sn, Ge)
    metals = ['Nb', 'Ti', 'Zr', 'V', 'Sn', 'Ge', 'Mo', 'Re', 'W']
    for combo in itertools.combinations(metals, 2):
        for x in np.arange(0.1, 1.0, 0.1):
            x_amt = round(x, 2)
            y_amt = round(1.0 - x, 2)
            candidates.add(f"{combo[0]}{x_amt}{combo[1]}{y_amt}")

    return list(candidates)

def main():
    os.makedirs("data", exist_ok=True)
    out_file = "data/work_queue.csv"
    
    print("Generating candidate compositions...")
    candidates = generate_combinatorial_space()
    print(f"Generated {len(candidates)} candidates.")
    
    # Optional: validate with pymatgen
    valid_candidates = []
    for c in candidates:
        try:
            # Just ensure it parses
            Composition(c)
            valid_candidates.append(c)
        except Exception:
            pass
            
    print(f"Validated {len(valid_candidates)} candidates.")
    
    # Initialize SQLite database for work queue
    import sqlite3
    import math
    os.makedirs("results", exist_ok=True)
    db_file = "results/candidates.db"
    
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Create work_queue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS work_queue (
            formula TEXT PRIMARY KEY,
            chunk_id INTEGER,
            status TEXT DEFAULT 'pending',
            claimed_at TIMESTAMP
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chunk_id ON work_queue (chunk_id)')
    
    # Also create predictions table so the DB is ready
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            formula TEXT PRIMARY KEY,
            predicted_tc REAL,
            uncertainty REAL
        )
    ''')
    
    # Insert candidates with chunk IDs (e.g., chunk size of 100)
    chunk_size = 100
    rows = []
    for i, c in enumerate(valid_candidates):
        chunk_id = math.floor(i / chunk_size)
        rows.append((c, chunk_id, 'pending', None))
        
    cursor.executemany('''
        INSERT OR IGNORE INTO work_queue (formula, chunk_id, status, claimed_at)
        VALUES (?, ?, ?, ?)
    ''', rows)
    
    conn.commit()
    conn.close()
            
    print(f"Wrote {len(rows)} candidates into {db_file} across {math.ceil(len(valid_candidates)/chunk_size)} chunks.")

if __name__ == "__main__":
    main()
