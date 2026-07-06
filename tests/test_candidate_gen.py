import pytest
from pymatgen.core import Structure, Lattice, Element
from superconductor.candidate_gen import (
    PhysicsAwareCandidateEngine, 
    ChargeNeutralityFilter, 
    IonicRadiusFilter, 
    ElectronegativityFilter,
    SymmetryFilter,
    GoldschmidtPerovskiteFilter,
    WyckoffPreservationFilter,
    BondValenceFilter
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
    filter_strict = ChargeNeutralityFilter()
    # Na1Cl1 is charge balanced (+1, -1)
    valid, conf, _ = filter_strict.validate(base_structure, base_structure)
    assert valid is True
    
    cand = base_structure.copy()
    cand.replace_species({"Na": "Mg"})
    valid, conf, _ = filter_strict.validate(base_structure, cand)
    assert valid is False

def test_ionic_radius_filter(base_structure):
    filter_radius = IonicRadiusFilter(max_diff=0.15)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf, _ = filter_radius.validate(base_structure, cand)
    assert valid is False

def test_electronegativity_filter(base_structure):
    filter_en = ElectronegativityFilter(max_diff=0.5)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf, _ = filter_en.validate(base_structure, cand)
    assert valid is True
    
    cand.replace_species({"K": "F"})
    valid, conf, _ = filter_en.validate(base_structure, cand)
    assert valid is False

def test_symmetry_filter(base_structure):
    filter_sym = SymmetryFilter(enforce_subgroup=True)
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf, _ = filter_sym.validate(base_structure, cand)
    assert valid is True
    
def test_goldschmidt_filter():
    filter_gold = GoldschmidtPerovskiteFilter()
    # Mock a perovskite SrTiO3
    lattice = Lattice.cubic(3.905)
    perov = Structure(
        lattice,
        ["Sr", "Ti", "O", "O", "O"],
        [
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.5]
        ]
    )
    valid, conf, _ = filter_gold.validate(perov, perov)
    assert valid is True
    
def test_wyckoff_filter(base_structure):
    filter_wyc = WyckoffPreservationFilter()
    cand = base_structure.copy()
    cand.replace_species({"Na": "K"})
    valid, conf, _ = filter_wyc.validate(base_structure, cand)
    assert valid is True

def test_bond_valence_filter(base_structure):
    filter_bv = BondValenceFilter(enabled=True, max_exact_atoms=100, fallback="heuristic")
    valid, conf, _ = filter_bv.validate(base_structure, base_structure)
    assert valid is True

def test_engine_generation(base_structure):
    config = {
        "candidate_generation": {
            "database_path": "data/tests_candidates.db",
            "enable_multiprocessing": False,
            "bond_valence": {
                "enabled": True,
                "max_exact_atoms": 100,
                "fallback": "heuristic"
            },
            "filters": {
                "charge_neutrality": True,
                "goldschmidt_perovskite": False,
                "wyckoff_preservation": True,
                "duplicate_detection": True
            }
        }
    }
    engine = PhysicsAwareCandidateEngine(config)
    
    # Test valid substitution (Na -> K)
    res = engine.generate(base_structure, "substitution", substitutions=[{"Na": "K"}])
    assert len(res) == 1
    assert "K" in res[0]["structure"].composition.as_dict()

    # Test vacancy generation
    res = engine.generate(base_structure, "vacancy", substitutions=["Na"])
    # May fail charge neutrality if not balanced, but let's check it doesn't crash
    assert isinstance(res, list)
