import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .materials_lake import MaterialsLake
from .material_entity import ExperimentRecord, ModelRecord, CheckpointRecord

logger = logging.getLogger(__name__)

class ExperimentRegistry:
    """
    Registry for tracking experiments, models, and checkpoints.
    Ensures scientific reproducibility and pipeline provenance.
    """
    def __init__(self, lake: Optional[MaterialsLake] = None):
        self.lake = lake or MaterialsLake()
        
    def register_experiment(self, config: Dict[str, Any], pipeline_version: str = "v1") -> ExperimentRecord:
        record = ExperimentRecord(
            pipeline_version=pipeline_version,
            config_snapshot=config
        )
        
        query = """
            INSERT INTO experiments (id, pipeline_version, git_commit, random_seed, gpu_info, config_snapshot, start_time, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.lake.execute_write(query, (
            record.id, record.pipeline_version, record.git_commit, record.random_seed, record.gpu_info, 
            json.dumps(record.config_snapshot), record.start_time, record.status
        ))
        
        logger.info(f"Registered Experiment: {record.id}")
        return record
        
    def end_experiment(self, exp_id: str, status: str = "COMPLETED"):
        end_time = datetime.utcnow().isoformat()
        # Event sourcing purists would append an ExperimentEnded event, 
        # but for top-level EXP metadata, an UPDATE is acceptable or we just insert an event.
        # Since we want append-only, we'll tolerate this single state update for EXP status,
        # or we could log an ExperimentStatusEvent. Let's just do an update for EXP status.
        query = "UPDATE experiments SET end_time = ?, status = ? WHERE id = ?"
        self.lake.execute_write(query, (end_time, status, exp_id))
        logger.info(f"Experiment {exp_id} ended with status: {status}")

    def register_model(self, name: str, architecture: str, version: str, training_dataset: str) -> ModelRecord:
        record = ModelRecord(name=name, architecture=architecture, version=version, training_dataset=training_dataset)
        query = "INSERT INTO models (id, name, architecture, version, training_dataset, timestamp) VALUES (?, ?, ?, ?, ?, ?)"
        self.lake.execute_write(query, (record.id, record.name, record.architecture, record.version, record.training_dataset, record.timestamp))
        logger.info(f"Registered Model: {record.id} ({name} {version})")
        return record
        
    def register_checkpoint(self, model_id: str, epoch: int, val_loss: float, weights_path: str) -> CheckpointRecord:
        record = CheckpointRecord(model_id=model_id, epoch=epoch, val_loss=val_loss, weights_path=weights_path)
        query = "INSERT INTO checkpoints (id, model_id, epoch, val_loss, weights_path, timestamp) VALUES (?, ?, ?, ?, ?, ?)"
        self.lake.execute_write(query, (record.id, record.model_id, record.epoch, record.val_loss, record.weights_path, record.timestamp))
        logger.info(f"Registered Checkpoint: {record.id} for Model {model_id}")
        return record
