import logging
import time
import datetime
from typing import List, Dict, Any, Optional
from superconductor.materials_lake import MaterialsLake
from superconductor.experiment import Experiment

logger = logging.getLogger(__name__)

class RecoveryManager:
    """
    Handles crashed experiments and intelligent failure recovery.
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    def check_for_crashed_experiments(self) -> List[Experiment]:
        """
        Finds any experiment that was running but did not gracefully complete or fail.
        """
        rows = self.lake.execute_read("SELECT id FROM experiments WHERE status = 'RUNNING'")
        experiments = []
        for row in rows:
            try:
                exp = Experiment.load_from_lake(self.lake, row["id"])
                experiments.append(exp)
            except Exception as e:
                logger.error(f"Failed to load crashed experiment {row['id']}: {e}")
                
        return experiments

    def prompt_recovery_action(self, experiments: List[Experiment]) -> Optional[Experiment]:
        """
        CLI prompt for recovering experiments. Returns the selected experiment to resume, if any.
        """
        if not experiments:
            return None
            
        print("\n" + "="*50)
        print("CRASH RECOVERY SYSTEM")
        print("="*50)
        print(f"Detected {len(experiments)} unfinished experiment(s).")
        
        for i, exp in enumerate(experiments):
            print(f"\n[{i+1}] Recovered Experiment:")
            print(f"    Experiment ID:   {exp.id}")
            print(f"    Last Stage:      {exp.current_stage}")
            print(f"    Completed:       {', '.join(exp.completed_stages)}")
            
            # Estimate checkpoint age
            chkpts = self.lake.execute_read(
                "SELECT timestamp FROM checkpoint_history WHERE experiment_id=? ORDER BY id DESC LIMIT 1",
                (exp.id,)
            )
            chkpt_age = "No checkpoints"
            if chkpts:
                try:
                    ts = datetime.datetime.fromisoformat(chkpts[0]["timestamp"])
                    age = datetime.datetime.now(datetime.timezone.utc) - ts
                    chkpt_age = f"{age.total_seconds() / 60:.1f} minutes ago"
                except Exception:
                    pass
            print(f"    Checkpoint Age:  {chkpt_age}")

        print("\nOptions:")
        print("  [1-N] Resume an experiment")
        print("  [R] Restart from scratch (creates new experiment)")
        print("  [A] Archive all crashed experiments and start new")
        
        while True:
            choice = input("\nSelect an action: ").strip().upper()
            if choice == 'R':
                return None
            elif choice == 'A':
                for exp in experiments:
                    self.lake.execute_write("UPDATE experiments SET status='ARCHIVED' WHERE id=?", (exp.id,))
                logger.info("Archived unfinished experiments.")
                return None
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(experiments):
                    selected = experiments[idx]
                    logger.info(f"Selected experiment {selected.id} for resumption.")
                    # Mark others as archived or leave them? We'll archive others for simplicity
                    for j, exp in enumerate(experiments):
                        if j != idx:
                            self.lake.execute_write("UPDATE experiments SET status='ARCHIVED' WHERE id=?", (exp.id,))
                    return selected
            print("Invalid choice.")

    def handle_failure(self, experiment: Experiment, stage_name: str, error: Exception) -> str:
        """
        Intelligent failure recovery logic.
        Returns the action to take: 'RETRY', 'CONTINUE', or 'ABORT'.
        """
        error_type = type(error).__name__
        error_msg = str(error)
        
        import traceback
        stack_trace = traceback.format_exc()
        
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        logger.error(f"Experiment {experiment.id} failed at {stage_name}: {error_type} - {error_msg}")
        
        # Determine Recovery Action
        action = "ABORT"
        
        if "CUDA out of memory" in error_msg:
            logger.warning("CUDA OOM detected. Recommend reducing batch size. Action: ABORT (requires config change)")
            action = "ABORT"
        elif "database is locked" in error_msg.lower() or "sqlite3.OperationalError" in error_type:
            logger.warning("SQLite Database Locked. Temporary contention issue. Action: RETRY")
            action = "RETRY"
        elif "Timeout" in error_type or "ConnectionError" in error_type:
            logger.warning("Network issue detected (API timeout). Action: RETRY")
            action = "RETRY"
        elif "pymatgen" in stack_trace.lower():
            logger.warning("Pymatgen parsing or structural error. Usually isolated to a single candidate. Action: CONTINUE")
            action = "CONTINUE"
            
        self.lake.execute_write("""
            INSERT INTO failure_history (
                experiment_id, stage_name, error_type, error_message, stack_trace, recovery_action, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment.id, stage_name, error_type, error_msg, stack_trace, action, now
        ))
        
        if action == "ABORT":
            experiment.status = "FAILED"
            experiment.save_to_lake(self.lake)
            
        return action
