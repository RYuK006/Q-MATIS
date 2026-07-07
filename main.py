from superconductor.compat import apply_platform_patches
apply_platform_patches()

import yaml
import logging
from superconductor.research_engine import ResearchExecutionEngine

def setup_global_logging(config):
    log_level_str = config.get('system', {}).get('log_level', 'INFO').upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    setup_global_logging(config)
    
    engine = ResearchExecutionEngine(config)
    engine.run()
