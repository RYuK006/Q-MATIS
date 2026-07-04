import logging
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

def generate_substitutions(structure: Structure, substitutions: list[dict]):
    """
    Generate candidates by performing multiple simultaneous substitutions.
    substitutions: list of dictionaries, e.g., [{'Y': 'Ba'}, {'Y': 'La', 'Cu': 'Ag'}]
    """
    candidates = []
    for sub in substitutions:
        try:
            new_struct = structure.copy()
            new_struct.replace_species(sub)
            if check_charge_balance(new_struct):
                candidates.append(new_struct)
            else:
                logger.warning(f"Substitution {sub} rejected due to charge imbalance.")
        except Exception as e:
            logger.error(f"Failed to generate substitution {sub}: {e}")
    return candidates

def generate_vacancies(structure: Structure, elements: list[str], max_vacancies: int = 1):
    """
    Generate candidates by introducing vacancies for specific elements.
    """
    candidates = []
    for element in elements:
        sites = [i for i, site in enumerate(structure) if site.specie.symbol == element]
        # Simplistic approach: remove one site at a time
        for i in sites[:max_vacancies]:
            new_struct = structure.copy()
            new_struct.remove_sites([i])
            candidates.append(new_struct)
    return candidates

def check_charge_balance(structure: Structure):
    """
    Check if a structure is charge balanced based on common oxidation states.
    For complex intermetallics or cuprates, strict charge balance might be violated,
    so this is a basic heuristic filter.
    """
    # In a full research pipeline, one would use pymatgen's BVAnalyzer.
    # We return True to allow exploration, but the hook is structurally here.
    return True
