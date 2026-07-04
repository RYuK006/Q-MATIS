from abc import ABC, abstractmethod
import sqlite3
import os
import logging
from typing import List, Dict, Any, Optional
from pymatgen.core import Structure

logger = logging.getLogger(__name__)

class BaseDataSource(ABC):
    """
    Abstract base class for all materials data sources.
    Enforces a strict interface for fetching, validating, and caching crystal structures.
    """
    def __init__(self, cache_dir: str = "data/cache", db_name: str = "materials_cache.db"):
        self.cache_dir = cache_dir
        self.db_path = os.path.join(cache_dir, db_name)
        os.makedirs(cache_dir, exist_ok=True)
        self._init_db()
        
    def _init_db(self):
        """Initializes the SQLite cache database for structures."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS structures (
                    id TEXT PRIMARY KEY,
                    source TEXT,
                    formula TEXT,
                    structure_json TEXT,
                    target_property TEXT,
                    metadata TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dataset_metadata (
                    source TEXT PRIMARY KEY,
                    version TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    checkpoint_index INTEGER DEFAULT 0
                )
            ''')
            conn.commit()

    @abstractmethod
    def fetch_data(self, limit: int = 0) -> List[Dict[str, Any]]:
        """
        Fetches data from the remote source or local CSV.
        Must return a list of dictionaries with 'id', 'structure', and 'target'.
        """
        pass

    def validate_structure(self, struct: Any) -> Optional[Structure]:
        """
        Validates that the given structure is a well-formed PyMatGen Structure.
        Returns the parsed Structure if valid, otherwise None.
        """
        try:
            if isinstance(struct, str):
                struct = Structure.from_str(struct, fmt="cif")
            if isinstance(struct, dict):
                struct = Structure.from_dict(struct)
            
            if not isinstance(struct, Structure):
                return None
            
            # Simple validation check: ensure it has sites and a lattice
            if len(struct.sites) == 0 or struct.lattice.volume <= 0:
                return None
                
            return struct
        except Exception as e:
            logger.debug(f"Structure validation failed: {e}")
            return None

    def validate_formula(self, formula: str) -> bool:
        """
        Basic chemical formula string validation.
        """
        if not formula or not isinstance(formula, str):
            return False
        # Very basic check, extendable via pymatgen Composition
        from pymatgen.core import Composition
        try:
            comp = Composition(formula)
            return len(comp.elements) > 0
        except Exception:
            return False

    def get_cached_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Retrieves cached structures for this data source.
        """
        import json
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, formula, structure_json, target_property, metadata FROM structures WHERE source=?", (source,))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                struct_json_str = row[2]
                struct = Structure.from_dict(json.loads(struct_json_str)) if struct_json_str else None
                
                target_val = row[3]
                if target_val:
                    try:
                        target_val = json.loads(target_val)
                    except json.JSONDecodeError:
                        try:
                            target_val = float(target_val)
                        except ValueError:
                            pass
                            
                results.append({
                    'id': row[0],
                    'formula': row[1],
                    'structure': struct,
                    'target': target_val,
                    'metadata': json.loads(row[4]) if row[4] else {}
                })
            return results

    def save_to_cache(self, source: str, data: List[Dict[str, Any]]):
        """
        Saves a batch of structures to the SQLite cache, ignoring duplicates via OR IGNORE.
        """
        import json
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for item in data:
                struct = item.get('structure')
                struct_json = json.dumps(struct.as_dict()) if struct else None
                meta_json = json.dumps(item.get('metadata', {}))
                
                target = item.get('target')
                if isinstance(target, dict):
                    target_str = json.dumps(target)
                else:
                    target_str = str(target)
                    
                cursor.execute('''
                    INSERT OR IGNORE INTO structures (id, source, formula, structure_json, target_property, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (item['id'], source, item['formula'], struct_json, target_str, meta_json))
            conn.commit()
            logger.info(f"Saved {len(data)} items to SQLite cache for source {source}")

    def update_checkpoint(self, source: str, version: str, index: int):
        """Updates the dataset version and checkpoint index for resumable downloads."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO dataset_metadata (source, version, checkpoint_index, last_updated)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(source) DO UPDATE SET 
                    version=excluded.version, 
                    checkpoint_index=excluded.checkpoint_index,
                    last_updated=CURRENT_TIMESTAMP
            ''', (source, version, index))
            conn.commit()

    def get_checkpoint(self, source: str) -> Dict[str, Any]:
        """Retrieves the current dataset version and checkpoint index."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT version, checkpoint_index, last_updated FROM dataset_metadata WHERE source=?", (source,))
            row = cursor.fetchone()
            if row:
                return {'version': row[0], 'index': row[1], 'last_updated': row[2]}
            return {'version': 'unknown', 'index': 0, 'last_updated': None}
