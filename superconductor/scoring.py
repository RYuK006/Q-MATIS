import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class PhysicsScoreCalculator:
    """
    Computes a composite physics score based on the confidence values 
    from various filters.
    """
    @staticmethod
    def calculate(filter_results: Dict[str, float]) -> float:
        if not filter_results:
            return 1.0
            
        score = 1.0
        # Simple geometric mean or product of confidences.
        # For simplicity, we just take the product.
        for filter_name, conf in filter_results.items():
            score *= conf
            
        return round(score, 4)

class CandidateRanker:
    """
    Ranks a list of candidate dictionaries based on the Phase B1 formula:
    Ranking = Predicted Tc * Physics Score * Confidence * Stability Score
    """
    @staticmethod
    def rank(candidates: list) -> list:
        for c in candidates:
            tc = c.get("predicted_tc", 0.0)
            phys = c.get("physics_score", 1.0)
            conf = c.get("uncertainty_confidence", 1.0) # (1.0 means no uncertainty penalty)
            stab = c.get("stability_score", 1.0)
            
            c["final_ranking_score"] = tc * phys * conf * stab
            
        return sorted(candidates, key=lambda x: x.get("final_ranking_score", 0.0), reverse=True)
