import pytest
import os
import torch
import torch.nn as nn
from superconductor.materials_lake import MaterialsLake
from superconductor.experiment import Experiment
from superconductor.checkpoint_manager import UniversalCheckpointManager

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)

@pytest.fixture
def test_lake(tmp_path):
    db_path = str(tmp_path / "test_lake.db")
    return MaterialsLake(db_path)

def test_checkpoint_manager_save_load(test_lake, tmp_path):
    exp = Experiment()
    exp.save_to_lake(test_lake)
    
    manager = UniversalCheckpointManager(test_lake)
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Modify model slightly
    with torch.no_grad():
        model.linear.weight.fill_(1.5)
        
    chkpt_dir = str(tmp_path / "checkpoints")
    
    filepath = manager.save_checkpoint(
        experiment=exp,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        epoch=5,
        step=100,
        metrics={"val_loss": 0.05},
        dataset_hash="hash123",
        stage_name="FINE_TUNING",
        checkpoint_dir=chkpt_dir
    )
    
    assert os.path.exists(filepath)
    
    # Load into new model
    new_model = DummyModel()
    new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01) # different LR
    
    epoch, step, metrics = manager.load_checkpoint(filepath, new_model, new_optimizer)
    
    assert epoch == 5
    assert step == 100
    assert metrics["val_loss"] == 0.05
    assert torch.allclose(new_model.linear.weight, torch.full_like(new_model.linear.weight, 1.5))
    
    # Check DB entry
    rows = test_lake.execute_read("SELECT * FROM checkpoint_history WHERE experiment_id = ?", (exp.id,))
    assert len(rows) == 1
    assert rows[0]["epoch"] == 5
