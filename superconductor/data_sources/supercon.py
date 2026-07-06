import os
import pandas as pd
import logging
from typing import List, Dict, Any
from superconductor.data_sources.base import BaseDataSource
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

class SuperConDataSource(BaseDataSource):
    """
    Data source for parsing the SuperCon CSV database.
    """
    def __init__(self, csv_path: str = "data/supercon.csv", cache_dir: str = "data/cache"):
        super().__init__(cache_dir=cache_dir)
        self.csv_path = csv_path

    def fetch_data(self, limit: int = 0) -> List[Dict[str, Any]]:
        cached_data = self.get_cached_data("SuperCon")
        if cached_data:
            logger.info(f"Loaded {len(cached_data)} items from cache for SuperCon.")
            if limit > 0:
                return cached_data[:limit]
            return cached_data

        if not os.path.exists(self.csv_path):
            logger.error(f"SuperCon CSV not found at {self.csv_path}")
            return []

        logger.info(f"Parsing SuperCon CSV from {self.csv_path}...")
        results = []
        try:
            df = pd.read_csv(self.csv_path)
            for idx, row in df.iterrows():
                # We assume SuperCon dataset has 'formula' and 'tc' and some structure format.
                # Since SuperCon historically doesn't have CIFs by default, it might be generated.
                # In this pipeline, if we have 'cif' or 'structure_json' we parse it.
                struct = None
                if 'cif' in row and pd.notna(row['cif']):
                    struct = self.validate_structure(row['cif'])
                    
                if not struct:
                    # Fallback or invalid structure
                    pass
                    
                formula = str(row.get('formula', 'Unknown'))
                if formula == 'Unknown':
                    formula = str(row.get('name', 'Unknown')) # fallback for supercon dataset column
                    
                tc = float(row.get('tc', row.get('Tc', 0.0)))
                
                results.append({
                    'id': f"supercon_{idx}",
                    'formula': formula,
                    'structure': struct,
                    'target': {'tc': tc},
                    'metadata': {'source': 'SuperCon', 'row_id': idx}
                })
                
                if limit > 0 and len(results) >= limit:
                    break
        except Exception as e:
            logger.error(f"Error parsing SuperCon data: {e}")
            
        if results:
            self.save_to_cache("SuperCon", results)
            
        return results
