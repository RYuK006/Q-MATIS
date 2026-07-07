import pytest
import os
from pymatgen.core import Structure, Lattice
from superconductor.materials_lake import MaterialsLake
from superconductor.material_registry import MaterialRegistry
from superconductor.experiment_registry import ExperimentRegistry
from superconductor.candidate_gen import PhysicsAwareCandidateEngine
from superconductor.state_manager import ResearchStateManager

@pytest.fixture
def test_lake(tmp_path):
    db_path = str(tmp_path / "test_lake.db")
    return MaterialsLake(db_path)

def test_pipeline_stage_resume(test_lake):
    state_mgr = ResearchStateManager(test_lake)
    exp_reg = ExperimentRegistry(test_lake)
    exp = exp_reg.register_experiment({"test": 1})
    exp_id = exp.id
    
    state_mgr.log_stage_start(exp_id, "PRETRAIN")
    # Verify in DB
    stages = test_lake.execute_read("SELECT * FROM stage_history WHERE experiment_id=?", (exp_id,))
    assert len(stages) == 1
    assert stages[0]["stage_name"] == "PRETRAIN"
    assert stages[0]["status"] == "RUNNING"
    
    # End stage
    state_mgr.log_stage_end(exp_id, "PRETRAIN", "COMPLETED")
    stages = test_lake.execute_read("SELECT * FROM stage_history WHERE experiment_id=?", (exp_id,))
    assert stages[0]["status"] == "COMPLETED"

def test_candidate_cursor_resume(test_lake):
    # Setup Engine
    registry = MaterialRegistry(test_lake)
    exp_reg = ExperimentRegistry(test_lake)
    exp = exp_reg.register_experiment({"test": 1})
    
    engine = PhysicsAwareCandidateEngine({}, registry, exp.id)
    
    lattice = Lattice.cubic(3.905)
    structure = Structure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5]
        ]
    )
    
    subs = [{"Sr": "Ba"}, {"Sr": "Ca"}, {"Ti": "Zr"}, {"Sr": "K"}]
    batch_id = "BATCH-TEST-1"
    
    # 1. Run full batch
    engine.generate(structure, "substitution", substitutions=subs, batch_id=batch_id)
    
    state_mgr = ResearchStateManager(test_lake)
    cursor = state_mgr.get_candidate_cursor(batch_id)
    assert cursor == len(subs) - 1 # 0-indexed, so 3
    
    # 2. Re-run with the same batch_id -> Should instantly skip everything
    # Just check that it doesn't crash and cursor remains the same
    engine.generate(structure, "substitution", substitutions=subs, batch_id=batch_id)
    assert state_mgr.get_candidate_cursor(batch_id) == len(subs) - 1

def test_partial_candidate_resume(test_lake):
    state_mgr = ResearchStateManager(test_lake)
    exp_reg = ExperimentRegistry(test_lake)
    exp = exp_reg.register_experiment({"test": 1})
    exp_id = exp.id
    
    batch_id = "BATCH-TEST-2"
    
    # Manually simulate that we crashed at index 1 (meaning 0 and 1 are processed)
    state_mgr.update_candidate_cursor(batch_id, exp_id, 4, 1)
    
    registry = MaterialRegistry(test_lake)
    engine = PhysicsAwareCandidateEngine({}, registry, exp_id)
    
    lattice = Lattice.cubic(3.905)
    structure = Structure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]
    )
    subs = [{"Sr": "Ba"}, {"Sr": "Ca"}, {"Ti": "Zr"}, {"Sr": "K"}]
    
    # Should only process indices 2 and 3 ("Zr" and "K")
    engine.generate(structure, "substitution", substitutions=subs, batch_id=batch_id)
    
    new_cursor = state_mgr.get_candidate_cursor(batch_id)
    assert new_cursor == 3
