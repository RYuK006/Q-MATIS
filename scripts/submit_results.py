import sqlite3
import argparse

def main():
    parser = argparse.ArgumentParser(description="Submit chunk results back to the canonical database")
    parser.add_argument("res_db", type=str, help="Path to the worker's results SQLite DB (e.g., res_chunk_1.db)")
    parser.add_argument("--main_db", type=str, default="results/candidates.db", help="Path to main canonical SQLite DB")
    args = parser.parse_args()

    # Connect to the main database
    conn = sqlite3.connect(args.main_db)
    cursor = conn.cursor()
    
    # Attach the worker database
    cursor.execute(f"ATTACH DATABASE '{args.res_db}' AS worker_db")
    
    # 1. Merge the predictions
    cursor.execute('''
        INSERT OR REPLACE INTO predictions (formula, predicted_tc, uncertainty)
        SELECT formula, predicted_tc, uncertainty FROM worker_db.predictions
    ''')
    
    # 2. Find which chunk these formulas belonged to and mark them done
    cursor.execute('''
        UPDATE work_queue 
        SET status = 'done' 
        WHERE formula IN (SELECT formula FROM worker_db.predictions)
    ''')
    
    # Count how many were updated
    cursor.execute("SELECT changes()")
    updated_formulas = cursor.fetchone()[0]
    
    # Optional: We can also verify which chunk_id was primarily completed
    cursor.execute('''
        SELECT chunk_id, COUNT(*) as cnt 
        FROM work_queue 
        WHERE formula IN (SELECT formula FROM worker_db.predictions)
        GROUP BY chunk_id
    ''')
    chunk_counts = cursor.fetchall()
    
    conn.commit()
    conn.close()
    
    print(f"Merged {updated_formulas} predictions into {args.main_db}.")
    if chunk_counts:
        for cid, cnt in chunk_counts:
            print(f" -> Marked {cnt} formulas from chunk {cid} as 'done'.")
    else:
        print("Warning: No matching formulas found in work_queue to mark as done.")

if __name__ == "__main__":
    main()
