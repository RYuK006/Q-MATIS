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
    def __init__(self, db_path: str = "data/qmatis_lake.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable Foreign Keys
            conn.execute("PRAGMA foreign_keys = ON;")
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
                    status TEXT,
                    dataset_sha256 TEXT,
                    dataset_version TEXT,
                    num_materials INTEGER,
                    preprocessing_version TEXT,
                    graph_generation_version TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS environment_metadata (
                    experiment_id TEXT PRIMARY KEY,
                    python_version TEXT,
                    torch_version TEXT,
                    cuda_version TEXT,
                    pyg_version TEXT,
                    os_info TEXT,
                    hardware_info TEXT,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
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
                    embedding_data BLOB,
                    encoder_architecture TEXT,
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

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    epoch INTEGER,
                    train_loss REAL,
                    val_loss REAL,
                    learning_rate REAL,
                    gradient_norm REAL,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generation_failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    parent_id TEXT,
                    generation_strategy TEXT,
                    exception_message TEXT,
                    stack_trace TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)

            # PHASE B3: EXPERIMENT EXECUTION ENGINE TRACKING
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS experiment_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    event_type TEXT,
                    description TEXT,
                    metadata_json TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stage_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    stage_name TEXT,
                    status TEXT,
                    start_time DATETIME,
                    end_time DATETIME,
                    duration_seconds REAL,
                    resume_point TEXT,
                    failure_reason TEXT,
                    retry_count INTEGER,
                    resource_usage_json TEXT,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS checkpoint_history (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    model_id TEXT,
                    stage_name TEXT,
                    epoch INTEGER,
                    step INTEGER,
                    dataset_hash TEXT,
                    config_snapshot TEXT,
                    metrics_json TEXT,
                    checkpoint_path TEXT,
                    checksum TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resource_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    cpu_usage_percent REAL,
                    ram_usage_mb REAL,
                    gpu_usage_percent REAL,
                    vram_usage_mb REAL,
                    disk_usage_mb REAL,
                    db_size_mb REAL,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS autosave_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    trigger_type TEXT,
                    saved_elements TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS resume_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    resume_action TEXT,
                    previous_stage TEXT,
                    cursor_data TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS failure_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    stage_name TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    stack_trace TEXT,
                    recovery_action TEXT,
                    timestamp DATETIME,
                    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
                )
            """)

            # CREATE INDEXES FOR OPTIMIZATION
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_materials_parent ON materials(parent_id)",
                "CREATE INDEX IF NOT EXISTS idx_materials_rejected ON materials(is_rejected)",
                "CREATE INDEX IF NOT EXISTS idx_materials_strategy ON materials(generation_strategy)",
                "CREATE INDEX IF NOT EXISTS idx_materials_created ON materials(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_properties_material ON properties(material_id)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_material ON predictions(material_id)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_experiment ON predictions(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_predictions_checkpoint ON predictions(checkpoint_id)",
                "CREATE INDEX IF NOT EXISTS idx_decision_material ON decision_history(material_id)",
                "CREATE INDEX IF NOT EXISTS idx_decision_experiment ON decision_history(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_physics_material ON physics_audits(material_id)",
                "CREATE INDEX IF NOT EXISTS idx_embeddings_material ON embeddings(material_id)",
                "CREATE INDEX IF NOT EXISTS idx_embeddings_prediction ON embeddings(prediction_id)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_source ON relationships(source_id)",
                "CREATE INDEX IF NOT EXISTS idx_relationships_target ON relationships(target_id)",
                "CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON training_metrics_history(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_failures_experiment ON generation_failures(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_checkpoints_model ON checkpoints(model_id)",
                
                # New B3 Indexes
                "CREATE INDEX IF NOT EXISTS idx_exp_events_exp ON experiment_events(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_stage_hist_exp ON stage_history(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_chkpt_hist_exp ON checkpoint_history(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_resource_hist_exp ON resource_history(experiment_id)",
                "CREATE INDEX IF NOT EXISTS idx_failure_hist_exp ON failure_history(experiment_id)",
            ]
            
            for idx_query in indexes:
                cursor.execute(idx_query)

            conn.commit()

    def execute_write(self, query: str, parameters: tuple = ()):
        with sqlite3.connect(self.db_path) as conn:
            # Enforce FK on writes as well
            conn.execute("PRAGMA foreign_keys = ON;")
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
