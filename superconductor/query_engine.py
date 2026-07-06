import logging
from typing import Dict, Any, List, Optional
from .materials_lake import MaterialsLake

logger = logging.getLogger(__name__)

class QueryBuilder:
    """
    Fluent interface for querying the Materials Lake.
    """
    def __init__(self, lake: MaterialsLake):
        self.lake = lake
        self.conditions = []
        self.params = []
        self.join_predictions = False
        
    def is_rejected(self, rejected: bool):
        self.conditions.append("m.is_rejected = ?")
        self.params.append(1 if rejected else 0)
        return self

    def has_tc_gt(self, tc: float):
        self.join_predictions = True
        self.conditions.append("p.predicted_tc > ?")
        self.params.append(tc)
        return self

    def generated_from(self, parent_id: str):
        self.conditions.append("m.parent_id = ?")
        self.params.append(parent_id)
        return self
        
    def has_element(self, element: str):
        # A simple LIKE query for demonstration; a true element index would be better
        self.conditions.append("m.reduced_formula LIKE ?")
        self.params.append(f"%{element}%")
        return self

    def filter_by_family(self, family: str):
        if family.lower() == "perovskite":
            # Just matching the generation strategy for now
            self.conditions.append("m.generation_strategy LIKE ?")
            self.params.append("%perovskite%")
        return self

    def execute(self) -> List[Dict[str, Any]]:
        query = "SELECT m.* FROM materials m "
        
        if self.join_predictions:
            query += " LEFT JOIN predictions p ON m.id = p.material_id "
            
        if self.conditions:
            query += " WHERE " + " AND ".join(self.conditions)
            
        # Group by material id to avoid duplicates if joining
        if self.join_predictions:
            query += " GROUP BY m.id"
            
        return self.lake.execute_read(query, tuple(self.params))
