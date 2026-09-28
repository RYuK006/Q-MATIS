import time
import sqlite3
import itertools
import math
import os

def run_scale_test():
    metals = [
        'Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga',
        'Rb','Sr','Y','Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','In','Sn',
        'Cs','Ba','La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu',
        'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi'
    ] # 60 metals

    print(f"Using {len(metals)} metals for combinatorial generation.")
    print("Generating ternary candidates A(x) B(y) C(1-x-y) with step 0.05...")
    
    start_time = time.time()
    
    candidates = []
    # (60 choose 3) = 34,220 combinations
    # For step=0.05, we have x=0.05..0.90, y=0.05..(0.95-x) -> 171 fractional pairs
    # Total combinations = 34,220 * 171 = 5,851,620
    # We will just generate exactly 1,000,000 and stop.
    
    count = 0
    for combo in itertools.combinations(metals, 3):
        if count >= 1000000:
            break
        for x_int in range(5, 95, 5):
            for y_int in range(5, 95 - x_int, 5):
                x = x_int / 100.0
                y = y_int / 100.0
                z = 1.0 - x - y
                candidates.append(f"{combo[0]}{x:.2f}{combo[1]}{y:.2f}{combo[2]}{z:.2f}")
                count += 1
                if count >= 1000000:
                    break
                    
    gen_time = time.time() - start_time
    print(f"Generated {len(candidates)} formula strings in {gen_time:.2f} seconds.")
    
    os.makedirs("results", exist_ok=True)
    db_file = "results/scale_test.db"
    if os.path.exists(db_file):
        os.remove(db_file)
        
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE work_queue (
            formula TEXT PRIMARY KEY,
            chunk_id INTEGER,
            status TEXT DEFAULT 'pending',
            claimed_at TIMESTAMP
        )
    ''')
    
    start_db_time = time.time()
    chunk_size = 100
    
    # We use a generator expression for memory efficiency and fast batching
    def row_generator():
        for i, c in enumerate(candidates):
            yield (c, i // chunk_size, 'pending', None)
            
    # Insert in bulk chunks of 100,000 for speed
    batch_size = 100000
    rows = []
    for i, c in enumerate(candidates):
        rows.append((c, i // chunk_size, 'pending', None))
        if len(rows) >= batch_size:
            cursor.executemany("INSERT INTO work_queue VALUES (?, ?, ?, ?)", rows)
            rows = []
    if rows:
        cursor.executemany("INSERT INTO work_queue VALUES (?, ?, ?, ?)", rows)
        
    conn.commit()
    db_time = time.time() - start_db_time
    print(f"Inserted 1,000,000 rows into SQLite in {db_time:.2f} seconds.")
    
    total_time = time.time() - start_time
    print(f"Total time for 1M candidate sweep -> DB: {total_time:.2f} seconds.")

if __name__ == "__main__":
    run_scale_test()
