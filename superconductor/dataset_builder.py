import logging
import csv
from typing import List, Dict, Any
from .query_engine import QueryBuilder
from .materials_lake import MaterialsLake

logger = logging.getLogger(__name__)

class DatasetBuilder:
    """
    Constructs datasets for foundation model training from the registry.
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake

    def build_csv_dataset(self, query: QueryBuilder, output_path: str):
        materials = query.execute()
        if not materials:
            logger.warning("Query returned no materials. Dataset will be empty.")
            return

        # Fetch predictions for these materials to include in CSV
        mat_ids = [m["id"] for m in materials]
        
        preds_query = f"SELECT material_id, predicted_tc, uncertainty, physics_score FROM predictions WHERE material_id IN ({','.join(['?']*len(mat_ids))})"
        preds_raw = self.lake.execute_read(preds_query, tuple(mat_ids))
        preds_map = {p["material_id"]: p for p in preds_raw}
        
        # Write to CSV
        fieldnames = list(materials[0].keys()) + ["predicted_tc", "uncertainty", "physics_score"]
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for m in materials:
                row = dict(m)
                pred = preds_map.get(m["id"], {})
                row["predicted_tc"] = pred.get("predicted_tc", "")
                row["uncertainty"] = pred.get("uncertainty", "")
                row["physics_score"] = pred.get("physics_score", "")
                writer.writerow(row)
                
        logger.info(f"Dataset successfully built at {output_path} with {len(materials)} records.")

