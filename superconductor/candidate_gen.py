import logging
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
from pymatgen.core import Structure, Element
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN

logger = logging.getLogger(__name__)

# ==========================================
# 1. GENERATORS
# ==========================================

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, structure: Structure, **kwargs) -> List[Structure]:
        pass

class SubstitutionGenerator(BaseGenerator):
    def generate(self, structure: Structure, substitutions: List[Dict[str, str]], **kwargs) -> List[Structure]:
        """
        Generates candidates via single/multi-site elemental substitutions.
        substitutions: list of dictionaries, e.g., [{'Y': 'Ba'}, {'Y': 'La', 'Cu': 'Ag'}]
        """
        candidates = []
        for sub in substitutions:
            try:
                new_struct = structure.copy()
                new_struct.replace_species(sub)
                candidates.append(new_struct)
            except Exception as e:
                logger.error(f"Failed to generate substitution {sub}: {e}")
        return candidates

class VacancyGenerator(BaseGenerator):
    def generate(self, structure: Structure, elements: List[str], max_vacancies: int = 1, **kwargs) -> List[Structure]:
        """
        Generates substoichiometric candidates by removing specific elements.
        """
        candidates = []
        for element in elements:
            sites = [i for i, site in enumerate(structure) if site.specie.symbol == element]
            # Simplistic approach: remove up to `max_vacancies` sites (one by one for now)
            for i in sites[:max_vacancies]:
                new_struct = structure.copy()
                new_struct.remove_sites([i])
                candidates.append(new_struct)
        return candidates

class AlloyGenerator(BaseGenerator):
    def generate(self, structure: Structure, alloys: List[Dict[str, Dict[str, float]]], **kwargs) -> List[Structure]:
        """
        Generates solid-solutions via partial substitutions.
        alloys: [{'Y': {'Y': 0.5, 'Ba': 0.5}}]
        """
        candidates = []
        for alloy in alloys:
            try:
                new_struct = structure.copy()
                new_struct.replace_species(alloy)
                candidates.append(new_struct)
            except Exception as e:
                logger.error(f"Failed to generate alloy {alloy}: {e}")
        return candidates

# ==========================================
# 2. PHYSICS FILTERS
# ==========================================

class BaseFilter(ABC):
    @abstractmethod
    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        """
        Returns (is_valid, confidence_score)
        confidence_score is between 0.0 and 1.0
        """
        pass

class ChargeNeutralityFilter(BaseFilter):
    def __init__(self, mode: str = "adaptive"):
        # modes: 'strict', 'mixed_valence', 'adaptive'
        self.mode = mode

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        comp = candidate.composition
        try:
            # Try to guess oxidation states
            guesses = comp.oxi_state_guesses()
            if guesses:
                # We found a valid charge balanced integer state
                return True, 1.0
        except Exception:
            pass
            
        if self.mode == "strict":
            logger.debug(f"Strict Mode: Rejected {comp.reduced_formula} due to charge imbalance.")
            return False, 0.0
        elif self.mode == "mixed_valence":
            # For cuprates/intermetallics, strict guesses might fail. Let's do a basic electronegativity sum check.
            # If all are highly electronegative or highly electropositive, reject.
            has_metal = any(e.is_metal for e in comp.elements)
            has_nonmetal = any(not e.is_metal for e in comp.elements)
            if has_metal and has_nonmetal:
                return True, 0.8
            return False, 0.0
        else: # adaptive
            return True, 0.5 # Allow it but with lower confidence

class IonicRadiusFilter(BaseFilter):
    def __init__(self, max_diff: float = 0.15):
        self.max_diff = max_diff

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        # Hume-Rothery rule estimation
        # We compare the average atomic/ionic radius of the original and candidate
        # A true implementation would compare the specific substituted sites.
        # For simplicity, we just compare overall composition radius averages.
        orig_rad = sum([e.average_ionic_radius or e.atomic_radius or 1.0 for e in original.composition.elements]) / len(original.composition.elements)
        cand_rad = sum([e.average_ionic_radius or e.atomic_radius or 1.0 for e in candidate.composition.elements]) / len(candidate.composition.elements)
        
        diff = abs(orig_rad - cand_rad) / orig_rad
        if diff <= self.max_diff:
            return True, 1.0 - (diff / self.max_diff)
        return False, 0.0

class ElectronegativityFilter(BaseFilter):
    def __init__(self, max_diff: float = 0.5):
        self.max_diff = max_diff

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        orig_en = sum([e.X for e in original.composition.elements if e.X is not None]) / len([e for e in original.composition.elements if e.X is not None])
        cand_en = sum([e.X for e in candidate.composition.elements if e.X is not None]) / len([e for e in candidate.composition.elements if e.X is not None])
        
        diff = abs(orig_en - cand_en)
        if diff <= self.max_diff:
            return True, 1.0 - (diff / self.max_diff)
        return False, 0.0

class SymmetryFilter(BaseFilter):
    def __init__(self, enforce_subgroup: bool = True):
        self.enforce_subgroup = enforce_subgroup

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        # Check if symmetry is drastically broken
        sg_orig = SpacegroupAnalyzer(original).get_space_group_number()
        sg_cand = SpacegroupAnalyzer(candidate).get_space_group_number()
        
        # If symmetry drops to 1 (P1), it means it's heavily distorted
        if self.enforce_subgroup and sg_cand == 1 and sg_orig != 1:
            return False, 0.0
            
        return True, 1.0 if sg_orig == sg_cand else 0.5

class CoordinationFilter(BaseFilter):
    def __init__(self):
        self.cnn = CrystalNN(distance_cutoffs=None, x_diff_weight=0.0)

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float]:
        # Just check if CrystalNN can successfully build a coordination shell
        # without throwing geometric exceptions due to atom crashes.
        try:
            _ = self.cnn.get_nn_info(candidate, 0)
            return True, 1.0
        except Exception:
            return False, 0.0

# ==========================================
# 3. ENGINE
# ==========================================

class PhysicsAwareCandidateEngine:
    def __init__(self, config: Dict[str, Any]):
        filter_conf = config.get("physics_filters", {})
        
        self.generators = {
            "substitution": SubstitutionGenerator(),
            "vacancy": VacancyGenerator(),
            "alloy": AlloyGenerator()
        }
        
        self.filters = [
            ChargeNeutralityFilter(mode=filter_conf.get("charge_neutrality_mode", "adaptive")),
            IonicRadiusFilter(max_diff=filter_conf.get("max_ionic_radius_diff", 0.15)),
            ElectronegativityFilter(max_diff=filter_conf.get("max_electronegativity_diff", 0.5)),
            SymmetryFilter(enforce_subgroup=filter_conf.get("enforce_symmetry_subgroup", True)),
            CoordinationFilter()
        ]

    def generate(self, base_structure: Structure, strategy: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Generates and screens candidates. Returns a list of dictionaries containing:
        - 'structure': The passed Structure
        - 'confidence': Overall physics confidence score (0-1)
        """
        generator = self.generators.get(strategy)
        if not generator:
            logger.error(f"Unknown generation strategy: {strategy}")
            return []
            
        raw_candidates = generator.generate(base_structure, **kwargs)
        
        valid_candidates = []
        for cand in raw_candidates:
            is_valid = True
            total_conf = 1.0
            
            for f in self.filters:
                passed, conf = f.validate(base_structure, cand)
                if not passed:
                    is_valid = False
                    break
                total_conf *= conf
                
            if is_valid:
                valid_candidates.append({
                    "structure": cand,
                    "confidence": total_conf
                })
                
        logger.info(f"Generated {len(raw_candidates)} candidates via {strategy}, {len(valid_candidates)} passed physics filters.")
        return valid_candidates
