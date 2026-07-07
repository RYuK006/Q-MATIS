import logging
import copy
import math
import uuid
import datetime
import random
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import concurrent.futures
from functools import lru_cache

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher

from .material_entity import MaterialEntity, PhysicsAuditRecord, DecisionRecord, PredictionRecord, gen_id
from .material_registry import MaterialRegistry
from .materials_lake import MaterialsLake
from .state_manager import ResearchStateManager
from .scoring import PhysicsScoreCalculator

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
        candidates = []
        for element in elements:
            sites = [i for i, site in enumerate(structure) if site.specie.symbol == element]
            for i in sites[:max_vacancies]:
                new_struct = structure.copy()
                new_struct.remove_sites([i])
                candidates.append(new_struct)
        return candidates

class AlloyGenerator(BaseGenerator):
    def generate(self, structure: Structure, alloys: List[Dict[str, Dict[str, float]]], **kwargs) -> List[Structure]:
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

class PhysicsFilter(ABC):
    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        """Returns (passed, confidence_score, rejection_reason)"""
        pass

class ChargeNeutralityFilter(PhysicsFilter):
    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        # Cheap heuristic
        try:
            guesses = candidate.composition.oxi_state_guesses()
            if guesses:
                return True, 1.0, ""
        except Exception:
            pass
        return False, 0.0, "Charge imbalance"

class OxidationStateValidationFilter(PhysicsFilter):
    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        # Fast electronegativity check as proxy for oxidation state sanity
        orig_elems = [e for e in original.composition.elements if e.X]
        cand_elems = [e for e in candidate.composition.elements if e.X]
        if not orig_elems or not cand_elems:
            return True, 1.0, ""
        orig_en = sum([e.X for e in orig_elems]) / len(orig_elems)
        cand_en = sum([e.X for e in cand_elems]) / len(cand_elems)
        if abs(orig_en - cand_en) <= 0.8:
            return True, 0.9, ""
        return False, 0.0, "Oxidation state incompatibility"

class IonicRadiusFilter(PhysicsFilter):
    def __init__(self, max_diff: float = 0.15):
        self.max_diff = max_diff

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        orig_rad = sum([e.average_ionic_radius or e.atomic_radius or 1.0 for e in original.composition.elements]) / len(original.composition.elements)
        cand_rad = sum([e.average_ionic_radius or e.atomic_radius or 1.0 for e in candidate.composition.elements]) / len(candidate.composition.elements)
        
        diff = abs(orig_rad - cand_rad) / orig_rad
        if diff <= self.max_diff:
            return True, 1.0 - (diff / self.max_diff), ""
        return False, 0.0, f"Ionic radius difference {diff:.2f} exceeds {self.max_diff}."

class ElectronegativityFilter(PhysicsFilter):
    def __init__(self, max_diff: float = 0.5):
        self.max_diff = max_diff

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        orig_elems = [e for e in original.composition.elements if e.X is not None]
        cand_elems = [e for e in candidate.composition.elements if e.X is not None]
        
        if not orig_elems or not cand_elems:
            return True, 1.0, ""
            
        orig_en = sum([e.X for e in orig_elems]) / len(orig_elems)
        cand_en = sum([e.X for e in cand_elems]) / len(cand_elems)
        
        diff = abs(orig_en - cand_en)
        if diff <= self.max_diff:
            return True, 1.0 - (diff / self.max_diff), ""
        return False, 0.0, f"Electronegativity diff {diff:.2f} exceeds {self.max_diff}."

class SymmetryFilter(PhysicsFilter):
    def __init__(self, enforce_subgroup: bool = True):
        self.enforce_subgroup = enforce_subgroup

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        sg_orig = SpacegroupAnalyzer(original).get_space_group_number()
        sg_cand = SpacegroupAnalyzer(candidate).get_space_group_number()
        
        if self.enforce_subgroup and sg_cand == 1 and sg_orig != 1:
            return False, 0.0, "Drastic symmetry breaking detected (dropped to P1)."
            
        return True, 1.0 if sg_orig == sg_cand else 0.5, ""

class GoldschmidtPerovskiteFilter(PhysicsFilter):
    def __init__(self, bounds: Tuple[float, float] = (0.75, 1.05)):
        self.bounds = bounds

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        comp = candidate.composition
        if len(comp.elements) != 3:
            return True, 1.0, "" # Not a classical ABX3, bypass
        
        elems = sorted(comp.elements, key=lambda e: e.X if e.X is not None else 0.0)
        X = elems[-1]
        A, B = elems[0], elems[1]
        if (A.average_ionic_radius or A.atomic_radius or 1.0) < (B.average_ionic_radius or B.atomic_radius or 1.0):
            A, B = B, A
            
        rA = A.average_ionic_radius or A.atomic_radius or 1.0
        rB = B.average_ionic_radius or B.atomic_radius or 1.0
        rX = X.average_ionic_radius or X.atomic_radius or 1.0
        
        t = (rA + rX) / (math.sqrt(2) * (rB + rX))
        if self.bounds[0] <= t <= self.bounds[1]:
            return True, 1.0, ""
        return False, 0.0, "Goldschmidt Tolerance"

class WyckoffPreservationFilter(PhysicsFilter):
    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        try:
            w_orig = SpacegroupAnalyzer(original).get_symmetry_dataset()["wyckoffs"]
            w_cand = SpacegroupAnalyzer(candidate).get_symmetry_dataset()["wyckoffs"]
            if w_orig == w_cand:
                return True, 1.0, ""
            return False, 0.0, "Wyckoff mismatch"
        except Exception:
            return False, 0.0, "Wyckoff mismatch"

class BondValenceFilter(PhysicsFilter):
    def __init__(self, enabled: bool = True, max_exact_atoms: int = 100, fallback: str = "heuristic"):
        self.enabled = enabled
        self.max_exact_atoms = max_exact_atoms
        self.fallback = fallback

    def validate(self, original: Structure, candidate: Structure) -> Tuple[bool, float, str]:
        if not self.enabled:
            return True, 1.0, ""
            
        num_atoms = len(candidate)
        if num_atoms > self.max_exact_atoms and self.fallback == "heuristic":
            # Heuristic check (volume sanity)
            vol_diff = abs(original.volume - candidate.volume) / original.volume
            if vol_diff < 0.2:
                return True, 0.7, ""
            return False, 0.0, "Bond valence (heuristic volume failure)"
            
        try:
            bva = BVAnalyzer()
            _ = bva.get_valences(candidate)
            return True, 1.0, ""
        except Exception:
            return False, 0.0, "Bond valence"


# Helper for multiprocessing
def _process_candidate(args):
    candidate = args['candidate']
    original = args['base_structure']
    filters = args['filters']
    strategy = args['strategy']
    sub_pathway = args['sub_pathway']
    exp_id = args.get('experiment_id', '')
    chk_id = args.get('checkpoint_id', '')
    
    passed_filters = []
    rejected_filter = ""
    confidences = {}
    is_valid = True
    
    for f in filters:
        passed, conf, reason = f.validate(original, candidate)
        if passed:
            passed_filters.append(f.name)
            confidences[f.name] = conf
        else:
            is_valid = False
            rejected_filter = reason
            break
            
    phys_score = PhysicsScoreCalculator.calculate(confidences) if is_valid else 0.0

    # Initialize the entity
    cand_id = gen_id("MAT")
    
    entity = MaterialEntity(
        id=cand_id,
        formula=candidate.composition.formula,
        reduced_formula=candidate.composition.reduced_formula,
        source="AI-Generated",
        parent_id=original.composition.reduced_formula,
        generation_strategy=strategy,
        is_rejected=not is_valid,
        structure_json=candidate.as_dict()
    )
    
    # Add physics audits
    for f in filters:
        status, conf, reason = f.validate(original, candidate)
        entity.physics_audits.append(PhysicsAuditRecord(
            experiment_id=exp_id,
            filter_name=f.name,
            status="Pass" if status else "Fail",
            score=conf,
            reason=reason if not status else ""
        ))
        
    # Removed fake prediction stub. Predictions should only be added by true model inference.
    
    # Add decision record
    entity.decisions.append(DecisionRecord(
        experiment_id=exp_id,
        action="Rejected" if not is_valid else "Generated",
        reason=rejected_filter if not is_valid else "Passed all filters",
        parameters={"strategy": strategy, "sub_pathway": sub_pathway},
        responsible_module="PhysicsAwareCandidateEngine"
    ))
    
    return entity


# ==========================================
# 3. ENGINE
# ==========================================

class PhysicsAwareCandidateEngine:
    def __init__(self, config: Dict[str, Any], registry: MaterialRegistry = None, experiment_id: str = "", checkpoint_id: str = "", encoder: Any = None, model: Any = None):
        cg_conf = config.get("candidate_generation", {})
        
        lake = registry.lake if registry else MaterialsLake(cg_conf.get("database_path", "data/qmatis_lake.db"))
        self.registry = registry or MaterialRegistry(lake)
        self.state_manager = ResearchStateManager(lake)
        self.use_mp = cg_conf.get("enable_multiprocessing", False)
        self.experiment_id = experiment_id
        self.checkpoint_id = checkpoint_id
        self.encoder = encoder
        self.model = model
        
        # Instantiate generators
        self.generators = {
            "substitution": SubstitutionGenerator(),
            "vacancy": VacancyGenerator(),
            "alloy": AlloyGenerator()
        }
        
        # Instantiate ordered filters
        filters_conf = cg_conf.get("filters", {})
        self.filters = []
        if filters_conf.get("charge_neutrality", True):
            self.filters.append(ChargeNeutralityFilter())
            self.filters.append(OxidationStateValidationFilter())
        if filters_conf.get("goldschmidt_perovskite", True):
            self.filters.append(GoldschmidtPerovskiteFilter())
        if filters_conf.get("wyckoff_preservation", True):
            self.filters.append(WyckoffPreservationFilter())
            
        bv_conf = cg_conf.get("bond_valence", {})
        self.filters.append(BondValenceFilter(
            enabled=bv_conf.get("enabled", True),
            max_exact_atoms=bv_conf.get("max_exact_atoms", 100),
            fallback=bv_conf.get("fallback", "heuristic")
        ))
        
        self.enable_duplicate_detection = filters_conf.get("duplicate_detection", True)
        self._seen_formulas = set()

    def generate(self, base_structure: Structure, strategy: str = "substitution", substitutions: List[Any] = None, batch_id: Optional[str] = None) -> List[Dict]:
        """
        Generates and filters candidates. Resumes automatically if batch_id is provided.
        """
        generator = self.generators.get(strategy)
        if not generator:
            logger.error(f"Unknown generation strategy: {strategy}")
            return []
            
        kwargs = {}
        if strategy == "substitution":
            kwargs["substitutions"] = substitutions
        elif strategy == "vacancy":
            kwargs["elements"] = substitutions
        elif strategy == "alloy":
            kwargs["alloys"] = substitutions
            
        try:
            raw_candidates = generator.generate(base_structure, **kwargs)
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            import traceback
            tb = traceback.format_exc()
            self.registry.lake.execute_write("""
                INSERT INTO generation_failures (experiment_id, parent_id, generation_strategy, exception_message, stack_trace, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (self.experiment_id, base_structure.composition.reduced_formula, strategy, str(e), tb, datetime.utcnow().isoformat()))
            return []
        
        last_processed = -1
        if batch_id:
            last_processed = self.state_manager.get_candidate_cursor(batch_id)
            if last_processed >= 0:
                logger.info(f"Resuming batch {batch_id} from index {last_processed + 1}")
        
        # Fast Duplicate Detection (Composition-based early pruning)
        unique_candidates = []
        sub_pathways = substitutions if substitutions else []
        for i, cand in enumerate(raw_candidates):
            if i <= last_processed:
                continue # Skip already processed
                
            form = cand.composition.reduced_formula
            if self.enable_duplicate_detection:
                if form in self._seen_formulas:
                    ent = MaterialEntity(
                        id=gen_id("MAT"),
                        formula=cand.composition.formula,
                        reduced_formula=form,
                        source="AI-Generated",
                        parent_id=base_structure.composition.reduced_formula,
                        generation_strategy=strategy,
                        is_rejected=True,
                    )
                    ent.decisions.append(DecisionRecord(
                        experiment_id=self.experiment_id,
                        action="Rejected", reason="Duplicate detected", parameters={}, responsible_module="DuplicateDetection"
                    ))
                    self.registry.register_material(ent)
                    if batch_id:
                        # Update cursor even for duplicates to avoid re-evaluating them on crash
                        self.state_manager.update_candidate_cursor(batch_id, self.experiment_id, len(raw_candidates), i)
                    continue
                self._seen_formulas.add(form)
            unique_candidates.append((i, cand))
            
        args_list = [
            {'candidate': cand, 'base_structure': base_structure, 'filters': self.filters, 'strategy': strategy, 'sub_pathway': str(sub_pathways), 'experiment_id': self.experiment_id, 'checkpoint_id': self.checkpoint_id, 'raw_index': idx}
            for idx, cand in unique_candidates
        ]
        
        valid_candidates = []
        
        # Sequential processing for exact cursor tracking (or dispatch async and track completed)
        # We will iterate through args_list. To properly update the cursor in real-time,
        # we will process sequentially in this snippet if batch_id is used, or batch update.
        
        # For simplicity and robust resume, we do sequential cursor updates here. 
        # (In a real massive cluster, you'd batch updates).
        processed_count = last_processed
        
        if self.use_mp and not batch_id:
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = list(executor.map(_process_candidate, args_list))
        else:
            results = []
            for arg in args_list:
                res = _process_candidate(arg)
                results.append(res)
                if batch_id:
                    # Update DB cursor (sync) using the absolute raw_index
                    self.state_manager.update_candidate_cursor(batch_id, self.experiment_id, len(raw_candidates), arg['raw_index'])
                
        for ent in results:
            # Optionally predict and encode if models are available
            if not ent.is_rejected and self.model and self.encoder:
                try:
                    # Construct graph, encode, and predict
                    from superconductor.graph import convert_structure_to_graph
                    import torch
                    
                    graph_data = convert_structure_to_graph(Structure.from_dict(ent.structure_json))
                    # Batch of 1
                    graph_data.batch = torch.zeros(graph_data.x.size(0), dtype=torch.long)
                    
                    with torch.no_grad():
                        latent_vector = self.encoder.encode(graph_data).cpu().numpy().flatten()
                        preds = self.model({"batch": graph_data}) # Simplified call
                        tc = float(preds.get("tc", [0.0])[0])
                    
                    pred_record = PredictionRecord(
                        experiment_id=self.experiment_id,
                        checkpoint_id=self.checkpoint_id,
                        predicted_tc=tc,
                        uncertainty=0.0,
                        physics_score=ent.physics_audits[-1].score if ent.physics_audits else 0.0,
                        stability_score=1.0
                    )
                    ent.predictions.append(pred_record)
                    self.registry.register_material(ent)
                    self.registry.save_latent_vector(ent.id, pred_record.id, self.experiment_id, latent_vector, encoder_architecture=str(self.encoder.__class__.__name__))
                except Exception as e:
                    logger.error(f"Failed to encode/predict candidate {ent.reduced_formula}: {e}")
                    self.registry.register_material(ent)
            else:
                if not ent.is_rejected:
                    # Accepted but no model to predict/encode
                    ent.decisions.append(DecisionRecord(
                        experiment_id=self.experiment_id,
                        action="Accepted",
                        reason="Passed filters, no model available for embedding",
                        parameters={},
                        responsible_module="PhysicsAwareCandidateEngine"
                    ))
                self.registry.register_material(ent)
            
            if not ent.is_rejected:
                valid_candidates.append(ent)
                tc_str = f"{ent.predictions[0].predicted_tc:.2f}K" if ent.predictions else "N/A"
                logger.info(f"Candidate: {ent.reduced_formula} | TC Pred: {tc_str}")
            else:
                logger.debug(f"Candidate Rejected: {ent.reduced_formula}")
                
        return valid_candidates

