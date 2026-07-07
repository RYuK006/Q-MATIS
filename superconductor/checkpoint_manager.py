import os
import uuid
import torch
import json
import logging
import datetime
import hashlib
import numpy as np
import random
from typing import Dict, Any, Tuple
from superconductor.materials_lake import MaterialsLake
from superconductor.experiment import Experiment

logger = logging.getLogger(__name__)

class UniversalCheckpointManager:
    """
    Manages robust state serialization to prevent data loss.
    Provides methods to save/restore models, optimizers, RNG seeds, and configuration.
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    def generate_checksum(self, filepath: str) -> str:
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def save_checkpoint(
        self,
        experiment: Experiment,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        scaler: Any,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        dataset_hash: str,
        stage_name: str,
        checkpoint_dir: str
    ) -> str:
        """
        Saves a highly rigorous checkpoint containing all required variables for exact resumption.
        """
        checkpoint_id = f"QMATIS-CHK-{uuid.uuid4().hex[:8]}"
        filepath = os.path.join(checkpoint_dir, f"{checkpoint_id}.pt")
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Capture precise RNG states for flawless resumption
        rng_states = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate()
        }
        if torch.cuda.is_available():
            rng_states["cuda"] = torch.cuda.get_rng_state_all()

        state_dict = {
            "experiment_id": experiment.id,
            "epoch": epoch,
            "step": step,
            "model_architecture": model.__class__.__name__ if model else "None",
            "model_state_dict": model.state_dict() if model and hasattr(model, 'state_dict') else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer and hasattr(optimizer, 'state_dict') else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler and hasattr(scheduler, 'state_dict') else None,
            "scaler_state_dict": scaler.state_dict() if scaler and hasattr(scaler, 'state_dict') else None,
            "metrics": metrics,
            "rng_states": rng_states,
            "dataset_hash": dataset_hash,
            "config_snapshot": experiment.config_snapshot
        }
        
        torch.save(state_dict, filepath)
        
        checksum = self.generate_checksum(filepath)
        
        # Log to lake
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metrics_json = json.dumps(metrics)
        config_json = json.dumps(experiment.config_snapshot)
        
        # Assuming the model has an ID, or generate one if None
        model_id = getattr(model, "id", f"MOD-{experiment.id}")
        
        self.lake.execute_write("""
            INSERT INTO checkpoint_history (
                id, experiment_id, model_id, stage_name, epoch, step, 
                dataset_hash, config_snapshot, metrics_json, 
                checkpoint_path, checksum, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            checkpoint_id, experiment.id, model_id, stage_name, epoch, step,
            dataset_hash, config_json, metrics_json, filepath, checksum, now
        ))
        
        logger.info(f"Saved rigorous checkpoint {checkpoint_id} at epoch {epoch}")
        return filepath

    def load_checkpoint(self, filepath: str, model: Any, optimizer: Any = None, scheduler: Any = None, scaler: Any = None, device: str = 'cpu') -> Tuple[int, int, Dict]:
        """
        Loads the checkpoint state back into the provided objects, including exact RNG seeds.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")
            
        logger.info(f"Loading rigorous checkpoint from {filepath}")
        state = torch.load(filepath, map_location=device, weights_only=False) # Needs to load non-tensors for RNG state
        
        # Integrity checks could be done here (e.g. comparing config snapshot)
        
        if model and state.get('model_state_dict'):
            model.load_state_dict(state['model_state_dict'], strict=True)
            
        if optimizer and state.get('optimizer_state_dict'):
            optimizer.load_state_dict(state['optimizer_state_dict'])
            
        if scheduler and state.get('scheduler_state_dict'):
            scheduler.load_state_dict(state['scheduler_state_dict'])
            
        if scaler and state.get('scaler_state_dict'):
            scaler.load_state_dict(state['scaler_state_dict'])
            
        rng_states = state.get('rng_states', {})
        if 'torch' in rng_states:
            torch.set_rng_state(rng_states['torch'])
        if 'numpy' in rng_states:
            np.random.set_state(rng_states['numpy'])
        if 'python' in rng_states:
            random.setstate(rng_states['python'])
        if 'cuda' in rng_states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states['cuda'])
            
        epoch = state.get('epoch', 0)
        step = state.get('step', 0)
        metrics = state.get('metrics', {})
        
        return epoch, step, metrics
