import sqlite3
import os
import csv
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CandidateDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    parent_material TEXT,
                    generated_formula TEXT,
                    strategy TEXT,
                    substitution_pathway TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    passed_filters TEXT,
                    rejected_filters TEXT,
                    physics_score REAL,
                    predicted_tc REAL,
                    uncertainty REAL,
                    stability_score REAL,
                    dft_status TEXT,
                    is_valid BOOLEAN
                )
            """)
            conn.commit()
            
    def insert_candidate(self, candidate_data: Dict[str, Any]):
        keys = ['id', 'parent_material', 'generated_formula', 'strategy', 'substitution_pathway', 
                'passed_filters', 'rejected_filters', 'physics_score', 'predicted_tc', 
                'uncertainty', 'stability_score', 'dft_status', 'is_valid']
                
        values = []
        for k in keys:
            val = candidate_data.get(k, None)
            if isinstance(val, list):
                val = ",".join(val)
            values.append(val)
            
        placeholders = ",".join(["?"] * len(keys))
        columns = ",".join(keys)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"INSERT OR REPLACE INTO candidates ({columns}) VALUES ({placeholders})", values)
            conn.commit()
            
    def get_rejection_analytics(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rejected_filters FROM candidates WHERE is_valid = 0")
            rows = cursor.fetchall()
            
        stats = {}
        for row in rows:
            if row[0]:
                reasons = row[0].split(",")
                for r in reasons:
                    r = r.strip()
                    stats[r] = stats.get(r, 0) + 1
        return stats
        
    def export_to_csv(self, csv_path: str):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates")
            rows = cursor.fetchall()
            columns = [description[0] for description in cursor.description]
            
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(rows)
        logger.info(f"Exported candidate provenance to {csv_path}")
