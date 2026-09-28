import os
import yaml
import logging
from superconductor.data_sources.build_dataset import DataOrchestrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resolve_structures():
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    if os.environ.get("MP_API_KEY"):
        if 'data_sources' not in base_config:
            base_config['data_sources'] = {}
        base_config['data_sources']['api_key'] = os.environ.get("MP_API_KEY")

    orchestrator = DataOrchestrator(base_config)
    dataset = orchestrator.build_dataset(limit=0)
    print(f"Successfully resolved {len(dataset)} items.")

if __name__ == "__main__":
    resolve_structures()
