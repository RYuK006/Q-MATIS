import pytest
import os
import sqlite3
import numpy as np
from superconductor.materials_lake import MaterialsLake
from superconductor.material_registry import MaterialRegistry
from superconductor.experiment_registry import ExperimentRegistry
from superconductor.material_entity import MaterialEntity, DecisionRecord, PredictionRecord, gen_id
from superconductor.query_engine import QueryBuilder
from superconductor.dataset_builder import DatasetBuilder

@pytest.fixture
def test_lake(tmp_path):
    db_path = str(tmp_path / "test_lake.db")
    lake = MaterialsLake(db_path)
    return lake

def test_experiment_registry(test_lake):
    exp_reg = ExperimentRegistry(test_lake)
    exp = exp_reg.register_experiment({"test": 123})
    assert exp.id.startswith("QMATIS-EXP-")
    
    mod = exp_reg.register_model("Test", "ALIGNN", "v1", "ds")
    assert mod.id.startswith("QMATIS-MOD-")
    
    chk = exp_reg.register_checkpoint(mod.id, 1, 0.1, "path")
    assert chk.id.startswith("QMATIS-CHK-")
    
    # Check if DB has them
    assert len(test_lake.execute_read("SELECT * FROM experiments WHERE id=?", (exp.id,))) == 1
    assert len(test_lake.execute_read("SELECT * FROM models WHERE id=?", (mod.id,))) == 1
    assert len(test_lake.execute_read("SELECT * FROM checkpoints WHERE id=?", (chk.id,))) == 1

def test_event_sourcing_append(test_lake):
    registry = MaterialRegistry(test_lake)
    exp_reg = ExperimentRegistry(test_lake)
    
    exp = exp_reg.register_experiment({"test": 1})
    mod = exp_reg.register_model("Test", "ALIGNN", "v1", "ds")
    chk = exp_reg.register_checkpoint(mod.id, 1, 0.1, "path")
    
    mat_id = gen_id("MAT")
    entity = MaterialEntity(
        id=mat_id,
        formula="SrTiO3",
        reduced_formula="SrTiO3",
        source="Imported"
    )
    
    # First prediction event
    p1 = PredictionRecord(experiment_id=exp.id, checkpoint_id=chk.id, predicted_tc=10.0)
    entity.predictions.append(p1)
    registry.register_material(entity)
    
    # Check DB
    preds = test_lake.execute_read("SELECT predicted_tc FROM predictions WHERE material_id=?", (mat_id,))
    assert len(preds) == 1
    assert preds[0]["predicted_tc"] == 10.0
    
    # Second prediction event (Simulating re-evaluation later)
    # Reconstruct the entity or just register another entity with the same ID but new prediction
    entity2 = MaterialEntity(
        id=mat_id,
        formula="SrTiO3",
        reduced_formula="SrTiO3",
        source="Imported"
    )
    p2 = PredictionRecord(experiment_id=exp.id, checkpoint_id=chk.id, predicted_tc=20.0)
    entity2.predictions.append(p2)
    
    # Registering again should append p2 without overwriting p1, and without failing on materials unique constraint
    registry.register_material(entity2)
    
    preds_after = test_lake.execute_read("SELECT predicted_tc FROM predictions WHERE material_id=? ORDER BY timestamp ASC", (mat_id,))
    assert len(preds_after) == 2
    assert preds_after[0]["predicted_tc"] == 10.0
    assert preds_after[1]["predicted_tc"] == 20.0

def test_embeddings_storage(test_lake):
    registry = MaterialRegistry(test_lake)
    mat_id = gen_id("MAT")
    pred_id = gen_id("PRD")
    exp_reg = ExperimentRegistry(test_lake)
    exp = exp_reg.register_experiment({"test": 1})
    exp_id = exp.id
    
    mod = exp_reg.register_model("Test", "ALIGNN", "v1", "ds")
    chk = exp_reg.register_checkpoint(mod.id, 1, 0.1, "path")
    
    entity = MaterialEntity(id=mat_id, formula="YBa2Cu3O7", reduced_formula="YBa2Cu3O7", source="Imported")
    registry.register_material(entity)
    
    # We also need a fake prediction in the DB for the prediction FK
    p1 = PredictionRecord(id=pred_id, experiment_id=exp.id, checkpoint_id=chk.id, predicted_tc=10.0)
    entity.predictions.append(p1)
    registry._append_predictions(mat_id, [p1])
    
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    emb = registry.save_latent_vector(mat_id, pred_id, exp_id, vec)
    
    assert emb.id.startswith("QMATIS-EMB-")
    rows = test_lake.execute_read("SELECT * FROM embeddings WHERE material_id = ?", (mat_id,))
    assert len(rows) == 1
    assert rows[0]["embedding_data"] == vec.tobytes()
