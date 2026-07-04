import numpy as np
from pymatgen.core import Element

def get_node_features(site):
    """
    Extract a rich set of atomic features for a node using pymatgen Element.
    Handles missing values gracefully by substituting 0.0 or a default.
    """
    specie = site.specie
    
    # Safe extraction helper
    def safe_get(attr, default=0.0):
        val = getattr(specie, attr, default)
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return val
        return float(val)
        
    def safe_prop(prop, default=0.0):
        try:
            val = getattr(specie, prop)
            if val is None:
                return default
            if isinstance(val, (list, tuple)):
                return val
            return float(val)
        except Exception:
            return default

    # Basic
    z = float(specie.Z)
    mass = safe_get('atomic_mass')
    en = safe_get('X', 0.0)  # Electronegativity
    
    # Radii
    cov_rad = safe_get('covalent_radius', 0.0)
    atm_rad = safe_get('atomic_radius', 0.0)
    ion_rad = safe_get('average_ionic_radius', 0.0)
    
    # Energy
    ea = safe_get('electron_affinity', 0.0)
    ie1 = safe_get('ionization_energies', [0.0])
    ie1 = ie1[0] if isinstance(ie1, list) and len(ie1) > 0 else 0.0
    
    # Periodic Table position
    group = safe_get('group', 0.0)
    row = safe_get('row', 0.0)
    
    # Block mapping (s: 0, p: 1, d: 2, f: 3)
    block_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
    block = block_map.get(specie.block, -1) if hasattr(specie, 'block') else -1
    
    # Valence
    valence = 0.0
    if hasattr(specie, 'full_electronic_structure'):
        val_shell = specie.full_electronic_structure[-1] if len(specie.full_electronic_structure) > 0 else (0, 's', 0)
        valence = float(val_shell[2])
    
    # Volume and Polarizability
    vol = safe_get('molar_volume', 0.0)
    polar = safe_get('polarizability', 0.0)
    
    # Mendeleev Number
    mendeleev = safe_get('mendeleev_no', 0.0)

    # Oxidation States
    ox_states = safe_get('common_oxidation_states', [0.0])
    max_ox = float(max(ox_states)) if ox_states else 0.0
    min_ox = float(min(ox_states)) if ox_states else 0.0

    features = [
        z, mass, en, cov_rad, atm_rad, ion_rad, 
        ea, float(ie1), group, row, float(block), 
        valence, vol, polar, mendeleev, max_ox, min_ox
    ]
    
    # Ensure no NaN
    features = [0.0 if np.isnan(f) else f for f in features]
    
    return features

def get_node_feature_dim():
    """Return the dimensionality of the node feature vector."""
    return 17
