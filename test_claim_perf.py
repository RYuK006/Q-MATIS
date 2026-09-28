import sqlite3
import time
import subprocess
import os

def run_claims(n=10):
    start = time.time()
    for _ in range(n):
        subprocess.run([".\\.venv312\\Scripts\\python.exe", "scripts/claim_chunk.py"], capture_output=True)
    end = time.time()
    return end - start

def main():
    db_path = "results/candidates.db"
    
    # 1. Reset DB with just 100 chunks
    print("Initializing DB with 100 chunks...")
    if os.path.exists(db_path):
        os.remove(db_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("CREATE TABLE work_queue (formula TEXT PRIMARY KEY, chunk_id INTEGER, status TEXT DEFAULT 'pending', claimed_at TIMESTAMP)")
    for i in range(100):
        c.execute("INSERT INTO work_queue (formula, chunk_id, status) VALUES (?, ?, ?)", (f"f_{i}", i, 'pending'))
    conn.commit()
    
    # 2. Time 10 claims before scale
    print("Running 10 claims BEFORE scale...")
    t_before = run_claims(10)
    print(f"Time for 10 claims (small DB): {t_before:.4f} seconds ({t_before/10:.4f} s/claim)")
    
    # 3. Insert 1,000,000 dummy rows
    print("\nInserting 1,000,000 dummy rows...")
    start_insert = time.time()
    rows = []
    batch_size = 100000
    for i in range(1000000):
        rows.append((f"dummy_{i}", 1000 + (i//100), 'pending'))
        if len(rows) >= batch_size:
            c.executemany("INSERT INTO work_queue (formula, chunk_id, status) VALUES (?, ?, ?)", rows)
            rows = []
    if rows:
        c.executemany("INSERT INTO work_queue (formula, chunk_id, status) VALUES (?, ?, ?)", rows)
    conn.commit()
    print(f"Insert done in {time.time()-start_insert:.2f} seconds.")
    
    # 4. Create Index and Time 10 claims after scale
    print("\nCreating index on (status, chunk_id)...")
    start_idx = time.time()
    c.execute("CREATE INDEX idx_status_chunk ON work_queue(status, chunk_id)")
    conn.commit()
    print(f"Index created in {time.time()-start_idx:.2f} seconds.")
    
    print("\nRunning 10 claims AFTER scale (with index)...")
    t_after = run_claims(10)
    print(f"Time for 10 claims (1M DB, Indexed): {t_after:.4f} seconds ({t_after/10:.4f} s/claim)")
    
    conn.close()

if __name__ == "__main__":
    main()
