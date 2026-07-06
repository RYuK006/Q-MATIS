import os
import pytest
from superconductor.data_sources.base import BaseDataSource
from superconductor.data_sources.supercon import SuperConDataSource
from superconductor.data_sources.mp import MPDataSource
from superconductor.data_sources.build_dataset import DataOrchestrator

def test_base_data_source_caching(tmp_path):
    cache_dir = str(tmp_path / "cache")
    # Using a stub subclass to instantiate BaseDataSource
    class DummySource(BaseDataSource):
        def fetch_data(self, limit=0):
            return []
            
    ds = DummySource(cache_dir=cache_dir)
    assert os.path.exists(ds.db_path)

def test_formula_validation(tmp_path):
    cache_dir = str(tmp_path / "cache")
    ds = SuperConDataSource(cache_dir=cache_dir, csv_path="dummy.csv")
    assert ds.validate_formula("H2O") == True
    assert ds.validate_formula("InvalidFormula!") == False
    assert ds.validate_formula("") == False

def test_orchestrator_initialization(tmp_path):
    cache_dir = str(tmp_path / "cache")
    config = {
        'data_sources': {
            'dataset_cache_dir': cache_dir,
            'api_key': 'test_key'
        }
    }
    orchestrator = DataOrchestrator(config)
    assert orchestrator.mp_ds.api_key == 'test_key'
    assert orchestrator.supercon_ds.cache_dir == cache_dir
