import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import os
import torch
import platform
import subprocess

from .materials_lake import MaterialsLake

logger = logging.getLogger(__name__)

class ResearchStateManager:
    """
    Provides multi-level resumability (Epochs, Pipeline Stages, Candidate Generations).
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    # Checkpointing is now handled by checkpoint_manager.py
        
    def save_environment_state(self, exp_id: str):
        """
        Records the current environment metadata to ensure reproducible research.
        """
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode('utf-8').strip()
        except Exception:
            git_commit = "unknown"
            
        env_vars = {
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "")
        }
        
        hardware_info = {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_properties": str(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else ""
        }
        
        self.lake.execute_write("""
            INSERT INTO environment_metadata (experiment_id, python_version, torch_version, cuda_version, git_commit, hardware_info, env_vars, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            exp_id,
            platform.python_version(),
            torch.__version__,
            torch.version.cuda if torch.cuda.is_available() else "None",
            git_commit,
            json.dumps(hardware_info),
            json.dumps(env_vars),
            datetime.utcnow().isoformat()
        ))
        logger.info(f"Saved environment state for experiment {exp_id}")

    # ==========================================
    # LEVEL 2: PIPELINE STAGES
    # ==========================================
    def log_stage_start(self, exp_id: str, stage: str):
        now = datetime.utcnow().isoformat()
        self.lake.execute_write("""
            INSERT INTO stage_history (experiment_id, stage_name, status, start_time)
            VALUES (?, ?, ?, ?)
        """, (exp_id, stage, "RUNNING", now))
        
    def log_stage_end(self, exp_id: str, stage: str, status: str = "COMPLETED", failure_reason: str = None):
        now = datetime.utcnow().isoformat()
        
        # Get start time to calculate duration
        rows = self.lake.execute_read("""
            SELECT id, start_time FROM stage_history 
            WHERE experiment_id = ? AND stage_name = ? AND status = 'RUNNING'
            ORDER BY id DESC LIMIT 1
        """, (exp_id, stage))
        
        duration = 0.0
        if rows:
            try:
                start_dt = datetime.fromisoformat(rows[0]["start_time"])
                end_dt = datetime.fromisoformat(now)
                duration = (end_dt - start_dt).total_seconds()
            except Exception:
                pass
                
            self.lake.execute_write("""
                UPDATE stage_history
                SET status = ?, end_time = ?, duration_seconds = ?, failure_reason = ?
                WHERE id = ?
            """, (status, now, duration, failure_reason, rows[0]["id"]))

    # ==========================================
    # LEVEL 3: CANDIDATE CURSORS
    # ==========================================
    def update_candidate_cursor(self, batch_id: str, exp_id: str, total: int, last_processed: int, status: str = "RUNNING"):
        now = datetime.utcnow().isoformat()
        rows = self.lake.execute_read("SELECT batch_id FROM generation_cursors WHERE batch_id = ?", (batch_id,))
        if rows:
            self.lake.execute_write("""
                UPDATE generation_cursors
                SET last_processed_index = ?, status = ?, last_updated = ?
                WHERE batch_id = ?
            """, (last_processed, status, now, batch_id))
        else:
            self.lake.execute_write("""
                INSERT INTO generation_cursors (batch_id, experiment_id, total_requested, last_processed_index, status, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (batch_id, exp_id, total, last_processed, status, now))
            
    def get_candidate_cursor(self, batch_id: str) -> int:
        """
        Returns the last processed index for a batch. Returns -1 if no cursor exists.
        """
        rows = self.lake.execute_read("SELECT last_processed_index FROM generation_cursors WHERE batch_id = ?", (batch_id,))
        if rows:
            return rows[0]["last_processed_index"]
        return -1

    def log_resume_action(self, exp_id: str, action: str, previous_stage: str, cursor_data: str):
        now = datetime.utcnow().isoformat()
        self.lake.execute_write("""
            INSERT INTO resume_history (experiment_id, resume_action, previous_stage, cursor_data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (exp_id, action, previous_stage, cursor_data, now))
