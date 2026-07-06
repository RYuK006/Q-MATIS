import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

from .materials_lake import MaterialsLake

logger = logging.getLogger(__name__)

class ResearchStateManager:
    """
    Provides multi-level resumability (Epochs, Pipeline Stages, Candidate Generations).
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    # ==========================================
    # LEVEL 1: EPOCH CHECKPOINTS
    # ==========================================
    def save_epoch_checkpoint(self, model: Any, optimizer: Any, scheduler: Any, epoch: int, metrics: Dict, path: str):
        """
        Placeholder for saving PyTorch checkpoints.
        Will be fully implemented when PyTorch models are integrated.
        """
        try:
            import torch
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict() if hasattr(model, 'state_dict') else None,
                'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                'metrics': metrics
            }
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save(checkpoint, path)
            logger.info(f"Saved epoch {epoch} checkpoint to {path}")
        except ImportError:
            logger.warning("torch not installed. Mocking epoch checkpoint save.")
            
    def load_epoch_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        Placeholder for loading PyTorch checkpoints.
        """
        try:
            import torch
            return torch.load(path)
        except ImportError:
            logger.warning("torch not installed. Mocking epoch checkpoint load.")
            return {'epoch': 0}

    # ==========================================
    # LEVEL 2: PIPELINE STAGES
    # ==========================================
    def update_pipeline_stage(self, exp_id: str, stage: str, completed_stages: List[str]):
        """
        Updates the macro-stage of the pipeline for a given experiment.
        """
        now = datetime.utcnow().isoformat()
        
        # Check if exists
        rows = self.lake.execute_read("SELECT experiment_id FROM experiment_states WHERE experiment_id = ?", (exp_id,))
        if rows:
            self.lake.execute_write("""
                UPDATE experiment_states 
                SET current_stage = ?, completed_stages = ?, last_updated = ?
                WHERE experiment_id = ?
            """, (stage, json.dumps(completed_stages), now, exp_id))
        else:
            self.lake.execute_write("""
                INSERT INTO experiment_states (experiment_id, current_stage, completed_stages, metrics, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (exp_id, stage, json.dumps(completed_stages), "{}", now))
            
    def get_pipeline_stage(self, exp_id: str) -> Dict[str, Any]:
        rows = self.lake.execute_read("SELECT * FROM experiment_states WHERE experiment_id = ?", (exp_id,))
        if rows:
            row = dict(rows[0])
            row["completed_stages"] = json.loads(row["completed_stages"])
            return row
        return {}

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
