import json
import datetime
import uuid
import platform
import os
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from superconductor.materials_lake import MaterialsLake

def generate_experiment_id() -> str:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:6]
    return f"EXP-{timestamp}-{short_uuid}"

@dataclass
class Experiment:
    """
    The fundamental unit of execution in Q-MATIS.
    Every artifact (models, materials, checkpoints, metrics) belongs to an Experiment.
    """
    id: str = field(default_factory=generate_experiment_id)
    created_at: str = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat())
    status: str = "CREATED"  # CREATED, RUNNING, PAUSED, COMPLETED, FAILED
    current_stage: str = "INIT"
    completed_stages: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # Environment Provenance
    python_version: str = platform.python_version()
    torch_version: str = torch.__version__
    cuda_version: str = torch.version.cuda if torch.cuda.is_available() else "N/A"
    os_info: str = f"{platform.system()} {platform.release()}"
    git_commit: str = "UNKNOWN"
    hardware_info: str = "UNKNOWN"
    random_seed: int = 42

    # Run Stats
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0

    def save_to_lake(self, lake: MaterialsLake):
        """Persists the experiment and its environment metadata to the Materials Lake."""
        config_json = json.dumps(self.config_snapshot)
        
        # Upsert Experiment
        lake.execute_write("""
            INSERT INTO experiments (
                id, pipeline_version, git_commit, random_seed, gpu_info, config_snapshot, 
                start_time, end_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                status=excluded.status,
                end_time=excluded.end_time
        """, (
            self.id, "v1", self.git_commit, self.random_seed, self.hardware_info, config_json,
            self.start_time, self.end_time, self.status
        ))
        
        # Upsert Environment Metadata
        lake.execute_write("""
            INSERT INTO environment_metadata (
                experiment_id, python_version, torch_version, cuda_version, os_info, hardware_info
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(experiment_id) DO NOTHING
        """, (
            self.id, self.python_version, self.torch_version, self.cuda_version,
            self.os_info, self.hardware_info
        ))

    @classmethod
    def load_from_lake(cls, lake: MaterialsLake, experiment_id: str) -> 'Experiment':
        rows = lake.execute_read("SELECT * FROM experiments WHERE id=?", (experiment_id,))
        if not rows:
            raise ValueError(f"Experiment {experiment_id} not found.")
        
        row = rows[0]
        env_rows = lake.execute_read("SELECT * FROM environment_metadata WHERE experiment_id=?", (experiment_id,))
        env_row = env_rows[0] if env_rows else {}
        
        exp = cls(
            id=row["id"],
            status=row["status"],
            config_snapshot=json.loads(row["config_snapshot"]) if row.get("config_snapshot") else {},
            git_commit=row["git_commit"],
            random_seed=row["random_seed"],
            start_time=row["start_time"],
            end_time=row["end_time"]
        )
        
        if env_row:
            exp.python_version = env_row.get("python_version", exp.python_version)
            exp.torch_version = env_row.get("torch_version", exp.torch_version)
            exp.cuda_version = env_row.get("cuda_version", exp.cuda_version)
            exp.os_info = env_row.get("os_info", exp.os_info)
            exp.hardware_info = env_row.get("hardware_info", exp.hardware_info)
            
        # Try to load stage history to reconstruct current state
        stages = lake.execute_read("SELECT stage_name, status FROM stage_history WHERE experiment_id=? ORDER BY id ASC", (experiment_id,))
        exp.completed_stages = [s["stage_name"] for s in stages if s["status"] == "COMPLETED"]
        if stages:
            last_stage = stages[-1]
            exp.current_stage = last_stage["stage_name"]
            
        return exp

    def record_event(self, lake: MaterialsLake, event_type: str, description: str, metadata: dict = None):
        """Log a timeline event for this experiment."""
        metadata_json = json.dumps(metadata) if metadata else "{}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        lake.execute_write("""
            INSERT INTO experiment_events (experiment_id, event_type, description, metadata_json, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (self.id, event_type, description, metadata_json, now))
