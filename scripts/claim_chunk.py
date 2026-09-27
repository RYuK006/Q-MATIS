import sqlite3
import argparse
import csv
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description="Claim a pending chunk from the work queue")
    parser.add_argument("--db", type=str, default="results/candidates.db", help="Path to SQLite database")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db, timeout=10.0)
    cursor = conn.cursor()

    # Use a single atomic UPDATE ... RETURNING to prevent race conditions
    now = datetime.now().isoformat()
    cursor.execute('''
        UPDATE work_queue 
        SET status = 'claimed', claimed_at = ?
        WHERE chunk_id = (
            SELECT chunk_id FROM work_queue 
            WHERE status = 'pending' 
            ORDER BY chunk_id ASC 
            LIMIT 1
        )
        RETURNING chunk_id
    ''', (now,))
    
    result = cursor.fetchone()
    if not result:
        print("No pending chunks found in the work queue. All done!")
        conn.close()
        return
        
    chunk_id = result[0]
    
    # Fetch formulas for this claimed chunk
    cursor.execute('''
        SELECT formula FROM work_queue 
        WHERE chunk_id = ?
    ''', (chunk_id,))
    
    formulas = cursor.fetchall()
    conn.commit()
    conn.close()
    
    # Write to a chunk CSV
    out_file = f"chunk_{chunk_id}.csv"
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['formula'])
        for (f_str,) in formulas:
            writer.writerow([f_str])
            
    print(f"Claimed chunk {chunk_id} containing {len(formulas)} formulas.")
    print(f"Wrote to {out_file}")
    print(f"Next step: python scripts/batch_scorer.py --input {out_file} --output res_chunk_{chunk_id}.db")

if __name__ == "__main__":
    main()
