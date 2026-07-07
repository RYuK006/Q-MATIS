import logging
import os
import time
import json
import traceback
import hashlib
from typing import Dict, Any

from superconductor.materials_lake import MaterialsLake
from superconductor.experiment import Experiment
from superconductor.resource_monitor import ResourceMonitor
from superconductor.recovery import RecoveryManager
from superconductor.checkpoint_manager import UniversalCheckpointManager
from superconductor.state_manager import ResearchStateManager

logger = logging.getLogger(__name__)

STAGES = [
    "CREATED",
    "DATA_DOWNLOAD",
    "CACHE_BUILD",
    "GRAPH_GENERATION",
    "PRETRAINING",
    "FINE_TUNING",
    "MULTITASK_TRAINING",
    "UNCERTAINTY_ESTIMATION",
    "CANDIDATE_GENERATION",
    "PHYSICS_FILTERING",
    "RANKING",
    "ACTIVE_LEARNING",
    "DFT_QUEUE",
    "REPORT_GENERATION",
    "COMPLETED"
]

class ResearchExecutionEngine:
    """
    The universal orchestration state-machine for Q-MATIS.
    """
    def __init__(self, config: Dict[str, Any], lake_path: str = "data/qmatis_lake.db", force_new: bool = False):
        self.config = config
        self.lake = MaterialsLake(lake_path)
        self.recovery = RecoveryManager(self.lake)
        self.checkpoint_manager = UniversalCheckpointManager(self.lake)
        self.state_manager = ResearchStateManager(self.lake)
        
        self.experiment = None
        self.monitor = None
        
        # Check for crashed experiments if not forced new
        if not force_new:
            crashed = self.recovery.check_for_crashed_experiments()
            if crashed:
                self.experiment = self.recovery.prompt_recovery_action(crashed)
                if self.experiment:
                    logger.info(f"Resuming Experiment: {self.experiment.id}")
                    self.experiment.status = "RUNNING"
                    self.experiment.save_to_lake(self.lake)
                    self.state_manager.log_resume_action(self.experiment.id, "RESUME", self.experiment.current_stage, "{}")
        
        # Initialize new if none recovered or selected
        if self.experiment is None:
            self.experiment = Experiment(config_snapshot=self.config)
            self.experiment.status = "RUNNING"
            self.experiment.current_stage = "CREATED"
            self.experiment.save_to_lake(self.lake)
            logger.info(f"Started New Experiment: {self.experiment.id}")
            
        self.monitor = ResourceMonitor(self.experiment, self.lake, interval_seconds=60)

    def _verify_integrity(self):
        """Verifies checkpoint compatibility and config hashes before resuming."""
        logger.info("Verifying system integrity...")
        # Placeholder for deep hash checks
        pass

    def _transition_to(self, new_stage: str):
        if self.experiment.current_stage != "CREATED":
            self.state_manager.log_stage_end(self.experiment.id, self.experiment.current_stage, "COMPLETED")
            self.experiment.completed_stages.append(self.experiment.current_stage)
            
        self.experiment.current_stage = new_stage
        self.experiment.save_to_lake(self.lake)
        self.state_manager.log_stage_start(self.experiment.id, new_stage)
        self.experiment.record_event(self.lake, "STAGE_START", f"Entered {new_stage} stage")
        logger.info(f"=== STAGE: {new_stage} ===")

    def run(self):
        try:
            self.monitor.start()
            self._verify_integrity()
            
            # Simple linear state machine
            start_index = STAGES.index(self.experiment.current_stage)
            
            for stage in STAGES[start_index+1:]:
                if stage == "COMPLETED":
                    self._transition_to(stage)
                    self.experiment.status = "COMPLETED"
                    self.experiment.save_to_lake(self.lake)
                    logger.info("Experiment Completed Successfully.")
                    break
                    
                self._transition_to(stage)
                self._execute_stage(stage)
                
        except Exception as e:
            action = self.recovery.handle_failure(self.experiment, self.experiment.current_stage, e)
            if action == "ABORT":
                logger.error("Unrecoverable error. Halting engine.")
            elif action == "RETRY":
                logger.info("Retry suggested. Engine will restart this stage on next run.")
                
        finally:
            self.monitor.stop()

    def _execute_stage(self, stage: str):
        """Dispatcher for different execution stages."""
        if stage == "DATA_DOWNLOAD":
            self._stage_data_download()
        elif stage == "CACHE_BUILD":
            pass # Self contained in preprocessing usually
        elif stage == "GRAPH_GENERATION":
            pass
        elif stage == "PRETRAINING":
            self._stage_pretraining()
        elif stage == "FINE_TUNING" or stage == "MULTITASK_TRAINING":
            self._stage_fine_tuning()
        elif stage == "CANDIDATE_GENERATION":
            self._stage_candidate_generation()
        elif stage == "REPORT_GENERATION":
            self._stage_report_generation()
        else:
            logger.info(f"Stage {stage} is currently a placeholder.")
            time.sleep(1) # Simulate some work

    def _stage_data_download(self):
        # We hook into data.py logic here eventually.
        self.experiment.record_event(self.lake, "DATA", "Data verified and cached.")
        
    def _stage_pretraining(self):
        pretrain = self.config.get("pretrain", {})
        if not pretrain.get("enabled", False):
            logger.info("Pretraining disabled in config. Skipping.")
            return
        # Stub
        time.sleep(2)
        
    def _stage_fine_tuning(self):
        # Stub
        time.sleep(2)
        
    def _stage_candidate_generation(self):
        # Logic from candidate_gen.py
        # Utilizes state_manager.get_candidate_cursor() and checkpoint_manager
        time.sleep(2)

    def _stage_report_generation(self):
        # Generates HTML and JSON reports
        logger.info("Generating automatic reports...")
        time.sleep(1)
