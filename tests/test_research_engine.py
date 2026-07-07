import pytest
import os
import yaml
from superconductor.materials_lake import MaterialsLake
from superconductor.research_engine import ResearchExecutionEngine
from superconductor.experiment import Experiment

@pytest.fixture
def test_lake(tmp_path):
    db_path = str(tmp_path / "test_lake.db")
    return MaterialsLake(db_path)

@pytest.fixture
def test_config():
    return {
        "system": {"random_seed": 42},
        "pretrain": {"enabled": False},
        "training": {},
        "model": {},
        "tasks": [{"name": "tc", "target_key": "tc"}]
    }

def test_research_engine_init(test_lake, test_config):
    # Test starting new experiment
    engine = ResearchExecutionEngine(test_config, lake_path=test_lake.db_path, force_new=True)
    assert engine.experiment is not None
    assert engine.experiment.status == "RUNNING"
    assert engine.experiment.current_stage == "CREATED"
    
def test_research_engine_resume(test_lake, test_config):
    # Create a crashed experiment
    exp = Experiment(current_stage="PRETRAINING", status="RUNNING")
    exp.save_to_lake(test_lake)
    
    # We can't easily test CLI input during `prompt_recovery_action` without mocking input.
    # But we can test that check_for_crashed_experiments finds it.
    from superconductor.recovery import RecoveryManager
    rm = RecoveryManager(test_lake)
    crashed = rm.check_for_crashed_experiments()
    assert len(crashed) == 1
    assert crashed[0].id == exp.id
