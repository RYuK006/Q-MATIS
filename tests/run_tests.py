import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import DiscoveryPipeline

def test_pipeline_initialization():
    print("Testing pipeline initialization...")
    pipeline = DiscoveryPipeline(config_path="../config.yaml")
    assert pipeline.config is not None
    assert 'model' in pipeline.config
    assert 'system' in pipeline.config
    print("Pipeline initialization OK")

if __name__ == "__main__":
    test_pipeline_initialization()
