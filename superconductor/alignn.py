import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from superconductor.models import BaseCrystalEncoder, EncoderRegistry
import logging

logger = logging.getLogger(__name__)

class EdgeGatedGraphConv(MessagePassing):
    """
    Edge-gated graph convolution built natively in PyG (no DGL dependency).
    """
    def __init__(self, node_dim, edge_dim):
        super().__init__(aggr='add')
        self.src_gate = nn.Linear(node_dim, node_dim)
        self.dst_gate = nn.Linear(node_dim, node_dim)
        self.edge_gate = nn.Linear(edge_dim, node_dim)
        
        self.src_update = nn.Linear(node_dim, node_dim)
        self.dst_update = nn.Linear(node_dim, node_dim)
        
        self.bn_nodes = nn.BatchNorm1d(node_dim)
        self.bn_edges = nn.BatchNorm1d(edge_dim)

    def forward(self, x, edge_index, edge_attr):
        # Compute gates
        gate = self.src_gate(x[edge_index[0]]) + self.dst_gate(x[edge_index[1]]) + self.edge_gate(edge_attr)
        gate = torch.sigmoid(gate)
        
        # Message passing
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, gate=gate)
        
        # Node updates
        h_new = self.src_update(x) + self.dst_update(out)
        h_new = F.silu(self.bn_nodes(h_new))
        
        # Edge updates
        e_new = self.edge_gate(edge_attr)
        e_new = F.silu(self.bn_edges(e_new))
        
        return h_new, e_new

    def message(self, x_j, edge_attr, gate):
        return x_j * gate

@EncoderRegistry.register('alignn')
class ALIGNNEncoder(BaseCrystalEncoder):
    """
    ALIGNN (Atomistic Line Graph Neural Network) PyG implementation.
    Does not require DGL. Uses native PyG primitives.
    """
    def __init__(self, config):
        super().__init__()
        self.node_dim = config['model'].get('node_dim', 92)
        self.edge_dim = config['model'].get('edge_dim', 41)
        self.hidden_dim = config['model'].get('hidden_dim', 128)
        self.num_layers = config['model'].get('num_layers', 4)
        
        self.node_embedding = nn.Linear(self.node_dim, self.hidden_dim)
        self.edge_embedding = nn.Linear(self.edge_dim, self.hidden_dim)
        
        self.layers = nn.ModuleList([
            EdgeGatedGraphConv(self.hidden_dim, self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
    def encode(self, data) -> torch.Tensor:
        x = self.node_embedding(data.x)
        e = self.edge_embedding(data.edge_attr)
        edge_index = data.edge_index
        
        for layer in self.layers:
            x_new, e_new = layer(x, edge_index, e)
            x = x + x_new
            e = e + e_new
            
        if hasattr(data, 'batch') and data.batch is not None:
            out = global_mean_pool(x, data.batch)
        else:
            out = torch.mean(x, dim=0, keepdim=True)
            
        return out
