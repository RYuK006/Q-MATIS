import sqlite3
import os

db_path = 'data/qmatis_lake.db'
if not os.path.exists(db_path):
    print(f'DB not found at {db_path}')
else:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for table in ['materials', 'decision_history']:
        try:
            cursor.execute(f'SELECT count(*) FROM {table}')
            print(f'Row count for {table}:', cursor.fetchone()[0])
        except Exception as e:
            print(f'Error querying {table}: {e}')
            
    try:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='materials'")
        print('Schema for materials:', cursor.fetchone()[0])
    except Exception as e:
        print('Error schema materials:', e)
        
    try:
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='decision_history'")
        print('Schema for decision_history:', cursor.fetchone()[0])
    except Exception as e:
        print('Error schema decision_history:', e)
