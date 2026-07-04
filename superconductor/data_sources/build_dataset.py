import os
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import time
from pymatgen.core import Composition

from superconductor.data_sources.supercon import SuperConDataSource
from superconductor.data_sources.mp import MPDataSource
from superconductor.data_sources.other_sources import JarvisDataSource, OqmdDataSource, AflowDataSource, AlexandriaDataSource

logger = logging.getLogger(__name__)

class DataOrchestrator:
    """
    Orchestrates multiple modular data sources, resolving structures for datasets 
    like SuperCon that may only provide chemical formulas.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = config.get('data_sources', {}).get('dataset_cache_dir', 'data/cache')
        self.mp_ds = MPDataSource(
            api_key=config.get('data_sources', {}).get('api_key'), 
            cache_dir=self.cache_dir
        )
        self.supercon_ds = SuperConDataSource(
            csv_path=config.get('data_sources', {}).get('supercon_csv', 'data/supercon.csv'),
            cache_dir=self.cache_dir
        )

    def build_dataset(self, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Builds the unified dataset. If a SuperCon entry lacks a structure, 
        it queries the Materials Project API as a fallback.
        Includes resumable checkpoints via the SQLite cache.
        """
        logger.info("Building unified dataset via DataOrchestrator...")
        
        # 1. Fetch from SuperCon (cached or fresh parsing)
        supercon_data = self.supercon_ds.fetch_data(limit=limit)
        
        # 2. Check which entries need structural resolution
        dataset = []
        needs_resolution = []
        
        for item in supercon_data:
            if item.get('structure'):
                dataset.append(item)
            else:
                needs_resolution.append(item)
                
        if not needs_resolution:
            logger.info(f"All {len(dataset)} items have structures.")
            return dataset[:limit] if limit > 0 else dataset
            
        logger.info(f"{len(needs_resolution)} SuperCon items require structural resolution from MP.")
        
        # 3. Resolve missing structures via Materials Project (in chunks)
        # We query the MPDataSource's cache first to avoid redundant API calls
        mp_cached = self.mp_ds.get_cached_data("MP")
        mp_cache_map = {}
        for row in mp_cached:
            try:
                red_f = Composition(row['formula']).reduced_formula
                mp_cache_map[red_f] = row['structure']
            except Exception:
                pass

        resolved = []
        from mp_api.client import MPRester
        
        if self.mp_ds.api_key:
            with MPRester(self.mp_ds.api_key) as mpr:
                chunk_size = 100
                for i in tqdm(range(0, len(needs_resolution), chunk_size), desc="Resolving structures"):
                    chunk = needs_resolution[i:i+chunk_size]
                    formulas_to_query = []
                    
                    for item in chunk:
                        if not self.supercon_ds.validate_formula(item['formula']):
                            continue
                        
                        try:
                            red_f = Composition(item['formula']).reduced_formula
                            if red_f in mp_cache_map:
                                item['structure'] = mp_cache_map[red_f]
                                dataset.append(item)
                            else:
                                formulas_to_query.append(item['formula'])
                        except Exception:
                            pass
                            
                    if not formulas_to_query:
                        continue
                        
                    # Query MP API for missing formulas
                    retries = 3
                    for attempt in range(retries):
                        try:
                            docs = mpr.summary.search(
                                formula=formulas_to_query,
                                is_stable=True,
                                fields=["material_id", "structure", "formula_pretty"]
                            )
                            # Update local map and save to cache
                            new_mp_items = []
                            for doc in docs:
                                struct = self.mp_ds.validate_structure(doc.structure)
                                if struct:
                                    red_f = Composition(doc.formula_pretty).reduced_formula
                                    mp_cache_map[red_f] = struct
                                    new_mp_items.append({
                                        'id': str(doc.material_id),
                                        'formula': doc.formula_pretty,
                                        'structure': struct,
                                        'target': {'formation_energy': 0.0}, # We don't have formation energy for pure resolution, but to keep schema consistent
                                        'metadata': {'source': 'MP'}
                                    })
                            if new_mp_items:
                                self.mp_ds.save_to_cache("MP", new_mp_items)
                            break
                        except Exception as e:
                            logger.warning(f"MP API Chunk failed: {e}. Retrying {attempt+1}/{retries}")
                            time.sleep(2 ** attempt)
                            
                    # Re-check chunk after query
                    for item in chunk:
                        if 'structure' not in item or not item['structure']:
                            try:
                                red_f = Composition(item['formula']).reduced_formula
                                if red_f in mp_cache_map:
                                    item['structure'] = mp_cache_map[red_f]
                                    dataset.append(item)
                                    resolved.append(item)
                            except Exception:
                                pass
                                
                    # Resumable save: update supercon cache with resolved structures
                    if resolved:
                        self.supercon_ds.save_to_cache("SuperCon", resolved)
                        resolved = []
        else:
            logger.error("No MP_API_KEY provided. Cannot resolve structures.")
            
        logger.info(f"Dataset resolution complete. {len(dataset)} items have valid structures.")
        if limit > 0:
            return dataset[:limit]
        return dataset

def build_dataset(supercon_path: str, cache_dir: str, limit: int = 0):
    """Legacy wrapper for backward compatibility with main.py"""
    config = {
        'data_sources': {
            'supercon_csv': supercon_path,
            'dataset_cache_dir': cache_dir,
            'api_key': os.environ.get("MP_API_KEY")
        }
    }
    orchestrator = DataOrchestrator(config)
    return orchestrator.build_dataset(limit=limit)
