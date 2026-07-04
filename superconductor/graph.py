import torch
import numpy as np
from pymatgen.core import Structure
import logging
from torch_geometric.utils import coalesce, to_undirected
from torch_geometric.data import Data

logger = logging.getLogger(__name__)

class GaussianDistance:
    """
    Expands distance with Gaussian basis functions.
    Provides a non-linear, rich representation of interatomic distances.
    """
    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        distances: (N_edges, 1) torch.Tensor
        Returns: (N_edges, len(filter)) torch.Tensor
        """
        distances = distances.view(-1, 1)
        filter_tensor = torch.tensor(self.filter, dtype=torch.float32).view(1, -1).to(distances.device)
        return torch.exp(-(distances - filter_tensor)**2 / self.var**2)

def structure_to_graph(structure: Structure, rbf_expander: GaussianDistance, radius: float = 8.0, max_num_nbr: int = 12):
    """
    Constructs a PyTorch Geometric graph from a pymatgen Structure.
    - Bidirectional edges.
    - Periodic Boundary Conditions (via pymatgen's get_all_neighbors).
    - RBF edge features.
    """
    from .features import get_node_features
    
    # 1. Node Features
    node_features = [get_node_features(site) for site in structure]
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # 2. Edge Features & Graph Connectivity
    all_neighbors = structure.get_all_neighbors(r=radius)
    
    edge_indices = []
    edge_distances = []
    
    isolated_atoms = 0
    for i, neighbors in enumerate(all_neighbors):
        if len(neighbors) == 0:
            isolated_atoms += 1
            continue
            
        neighbors = sorted(neighbors, key=lambda x: x.nn_distance)
        for neighbor in neighbors[:max_num_nbr]:
            j = neighbor.index
            dist = neighbor.nn_distance
            
            edge_indices.append([i, j])
            edge_distances.append(dist)
            
    if isolated_atoms > 0:
        logger.warning(f"Structure {structure.formula} has {isolated_atoms} isolated atoms (no neighbors within {radius} Å).")

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr_raw = torch.tensor(edge_distances, dtype=torch.float32)
        
        # Expand distances with RBF
        edge_attr = rbf_expander.expand(edge_attr_raw)
        
        # Ensure graph is strictly bidirectional and remove duplicates
        edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=len(structure))
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes=len(structure), reduce="mean")
        
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(rbf_expander.filter)), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
