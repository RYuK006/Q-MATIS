import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import DiscoveryPipeline

def test_pipeline_initialization():
    pipeline = DiscoveryPipeline(config_path="config.yaml")
    assert pipeline.config is not None
    assert 'model' in pipeline.config
    assert 'system' in pipeline.config
