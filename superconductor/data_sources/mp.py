import os
import logging
from typing import List, Dict, Any
from mp_api.client import MPRester
from superconductor.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)

class MPDataSource(BaseDataSource):
    """
    Data source for fetching bulk structures from the Materials Project.
    """
    def __init__(self, api_key: str = None, cache_dir: str = "data/cache"):
        super().__init__(cache_dir=cache_dir)
        self.api_key = api_key or os.environ.get("MP_API_KEY")
        if not self.api_key:
            logger.warning("No MP_API_KEY provided. MPDataSource will only be able to serve cached data.")

    def fetch_data(self, limit: int = 0) -> List[Dict[str, Any]]:
        cached_data = self.get_cached_data("MP")
        if cached_data:
            logger.info(f"Loaded {len(cached_data)} items from cache for Materials Project.")
            if limit > 0:
                return cached_data[:limit]
            return cached_data

        if not self.api_key:
            logger.error("Cannot fetch new data from Materials Project without an API key.")
            return []

        logger.info("Fetching data from Materials Project via API...")
        results = []
        try:
            with MPRester(self.api_key) as mpr:
                # We'll fetch basic stable materials for pretraining. 
                # This could be paginated/resumable in a more advanced implementation.
                docs = mpr.summary.search(
                    is_stable=True,
                    fields=["material_id", "structure", "formula_pretty", "formation_energy_per_atom"]
                )
                
                for doc in docs:
                    struct = self.validate_structure(doc.structure)
                    if struct:
                        results.append({
                            'id': str(doc.material_id),
                            'formula': doc.formula_pretty,
                            'structure': struct,
                            'target': {'formation_energy': doc.formation_energy_per_atom},
                            'metadata': {'source': 'MP'}
                        })
                    
                    if limit > 0 and len(results) >= limit:
                        break
        except Exception as e:
            logger.error(f"Error fetching from Materials Project: {e}")
            
        if results:
            self.save_to_cache("MP", results)
            
        return results
