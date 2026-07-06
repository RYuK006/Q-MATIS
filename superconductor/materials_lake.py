import sqlite3
import os
import json
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class MaterialsLake:
    """
    Storage layer for the Materials Knowledge Graph & Scientific Memory System.
    Strictly event-sourced / append-only.
    """
    def __init__(self, db_path: str = "data/qmatis_lake.db", embeddings_dir: str = "data/embeddings"):
        self.db_path = db_path
        self.embeddings_dir = embeddings_dir
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        os.makedirs(os.path.abspath(self.embeddings_dir), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # CORE ENTITIES
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    pipeline_version TEXT,
                    git_commit TEXT,
                    random_seed INTEGER,
                    gpu_info TEXT,
                    config_snapshot TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    status TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    architecture TEXT,
                    version TEXT,
                    training_dataset TEXT,
                    timestamp DATETIME
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id TEXT PRIMARY KEY,
                    model_id TEXT,
                    epoch INTEGER,
                    val_loss REAL,
                    weights_path TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(model_id) REFERENCES models(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    id TEXT PRIMARY KEY,
                    formula TEXT,
                    reduced_formula TEXT,
                    source TEXT,
                    parent_id TEXT,
                    generation_strategy TEXT,
                    structure_json TEXT,
                    metadata_json TEXT,
                    is_rejected BOOLEAN,
                    created_at DATETIME
                )
            """)
            
            # EVENT HISTORY (Append-Only)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS properties (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    material_id TEXT,
                    property_name TEXT,
                    value TEXT,
                    unit TEXT,
                    property_type TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(material_id) REFERENCES materials(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY,
                    material_id TEXT,
                    experiment_id TEXT,
                    checkpoint_id TEXT,
                    predicted_tc REAL,
                    uncertainty REAL,
                    physics_score REAL,
                    stability_score REAL,
                    timestamp DATETIME,
                    FOREIGN KEY(material_id) REFERENCES materials(id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY(checkpoint_id) REFERENCES checkpoints(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_history (
                    id TEXT PRIMARY KEY,
                    material_id TEXT,
                    experiment_id TEXT,
                    action TEXT,
                    reason TEXT,
                    parameters TEXT,
                    responsible_module TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(material_id) REFERENCES materials(id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS physics_audits (
                    id TEXT PRIMARY KEY,
                    material_id TEXT,
                    experiment_id TEXT,
                    filter_name TEXT,
                    status TEXT,
                    score REAL,
                    reason TEXT,
                    threshold REAL,
                    intermediate_values TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(material_id) REFERENCES materials(id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id TEXT PRIMARY KEY,
                    material_id TEXT,
                    experiment_id TEXT,
                    prediction_id TEXT,
                    dimension INTEGER,
                    embedding_path TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(material_id) REFERENCES materials(id),
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id),
                    FOREIGN KEY(prediction_id) REFERENCES predictions(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT,
                    target_id TEXT,
                    relationship_type TEXT,
                    metadata TEXT,
                    FOREIGN KEY(source_id) REFERENCES materials(id),
                    FOREIGN KEY(target_id) REFERENCES materials(id)
                )
            """)
            
            # RESEARCH STATE TRACKING (Level 2 & 3 Resumability)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_states (
                    experiment_id TEXT PRIMARY KEY,
                    current_stage TEXT,
                    completed_stages TEXT,
                    metrics TEXT,
                    last_updated DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_cursors (
                    batch_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    total_requested INTEGER,
                    last_processed_index INTEGER,
                    status TEXT,
                    last_updated DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            conn.commit()

    def execute_write(self, query: str, parameters: tuple = ()):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            conn.commit()
            return cursor.lastrowid

    def execute_read(self, query: str, parameters: tuple = ()) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query, parameters)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
