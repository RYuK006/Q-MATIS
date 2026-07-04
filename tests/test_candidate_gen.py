import pytest
from pymatgen.core import Structure, Lattice, Element
from superconductor.candidate_gen import (
    PhysicsAwareCandidateEngine, 
    ChargeNeutralityFilter, 
    IonicRadiusFilter, 
    ElectronegativityFilter,
    SymmetryFilter
)

@pytest.fixture
def base_structure():
    # Simple NaCl cubic structure
    lattice = Lattice.cubic(5.64)
    structure = Structure(
        lattice, 
        ["Na", "Na", "Na", "Na", "Cl", "Cl", "Cl", "Cl"], 
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5],
            [0.5, 0.5, 0.5]
        ]
    )
    return structure

def test_charge_neutrality_filter(base_structure):
    # Na1Cl1 is charge balanced (+1, -1)
    filter_strict = ChargeNeutralityFilter(mode="strict")
    valid, conf = filter_strict.validate(base_structure, base_structure)
    assert valid is True
    assert conf == 1.0
    
    # Introduce an aliovalent substitution (Na -> Mg) => MgCl (imbalanced +2, -1)
    cand = base_structure.copy()
    cand.replace_species({"Na": "Mg"})
    valid, conf = filter_strict.validate(base_structure, cand)
    assert valid is False # Strict should fail

def test_ionic_radius_filter(base_structure):
    filter_radius = IonicRadiusFilter(max_diff=0.15)
    
    # Isovalent substitution Na -> K (size mismatch)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    
    valid, conf = filter_radius.validate(base_structure, cand)
    # Na ionic radius vs K ionic radius is a huge difference, should fail 15% rule
    assert valid is False

def test_electronegativity_filter(base_structure):
    # Na (0.93) -> K (0.82) is delta 0.11
    filter_en = ElectronegativityFilter(max_diff=0.5)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf = filter_en.validate(base_structure, cand)
    assert valid is True
    
    # Na (0.93) -> F (3.98) is massive
    cand.replace_species({"K": "F"})
    valid, conf = filter_en.validate(base_structure, cand)
    assert valid is False

def test_symmetry_filter(base_structure):
    filter_sym = SymmetryFilter(enforce_subgroup=True)
    
    # A symmetric substitution Na -> K should preserve Fm-3m (225)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf = filter_sym.validate(base_structure, cand)
    assert valid is True
    
def test_engine_generation(base_structure):
    config = {
        "physics_filters": {
            "charge_neutrality_mode": "adaptive",
            "max_ionic_radius_diff": 0.5, # relax to pass
            "max_electronegativity_diff": 0.5,
            "enforce_symmetry_subgroup": True
        }
    }
    engine = PhysicsAwareCandidateEngine(config)
    
    # Test valid substitution (Na -> K)
    # In adaptive mode, charge passes. Ionic radius diff is relaxed to 50%.
    res = engine.generate(base_structure, "substitution", substitutions=[{"Na": "K"}])
    assert len(res) == 1
    assert "K" in res[0]["structure"].composition.as_dict()

    # Test vacancy generation
    res = engine.generate(base_structure, "vacancy", elements=["Na"], max_vacancies=1)
    assert len(res) == 1
    assert res[0]["structure"].composition.as_dict()["Na"] == 3 # removed one Na
