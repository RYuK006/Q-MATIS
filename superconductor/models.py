import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseCrystalEncoder(nn.Module, ABC):
    """
    Abstract base class for all crystal graph encoders.
    """
    @abstractmethod
    def encode(self, data) -> torch.Tensor:
        """
        Takes a torch_geometric Data object and returns a fixed-size embedding.
        """
        pass
        
    def forward(self, data):
        return self.encode(data)

class EncoderRegistry:
    """Registry pattern for BaseCrystalEncoders."""
    _registry = {}
    
    @classmethod
    def register(cls, name):
        def wrapper(encoder_class):
            cls._registry[name.lower()] = encoder_class
            return encoder_class
        return wrapper
        
    @classmethod
    def build(cls, name, config):
        name = name.lower()
        if name == 'alignn' and 'alignn' not in cls._registry:
            try:
                import superconductor.alignn # Registers ALIGNNEncoder
            except Exception as e:
                logger.warning(f"Failed to load ALIGNN: {e}")
                
        if name not in cls._registry:
            raise ValueError(f"Unknown encoder '{name}'. Registered: {list(cls._registry.keys())}")
        return cls._registry[name](config)

class ModernCGCNNLayer(MessagePassing):
    """
    Modernized Crystal Graph Convolutional Neural Network Layer.
    Includes BatchNorm and optional Residual connections.
    """
    def __init__(self, node_dim, edge_dim, use_batch_norm=True, use_residual=True):
        super().__init__(aggr='add')
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        
        self.fc_msg = nn.Linear(2 * node_dim + edge_dim, node_dim)
        self.fc_update = nn.Linear(2 * node_dim, node_dim)
        
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(node_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.use_residual:
            out = out + x
        return out

    def message(self, x_i, x_j, edge_attr):
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return F.silu(self.fc_msg(tmp))

    def update(self, aggr_out, x):
        tmp = torch.cat([x, aggr_out], dim=-1)
        out = F.silu(self.fc_update(tmp))
        if self.use_batch_norm:
            out = self.bn(out)
        return out

@EncoderRegistry.register('cgcnn')
class CGCNNEncoder(BaseCrystalEncoder):
    """
    CGCNN implementation of the BaseCrystalEncoder interface.
    """
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get('model', {})
        node_dim = model_cfg['node_dim']
        edge_dim = model_cfg['edge_dim']
        self.hidden_dim = model_cfg['hidden_dim']
        num_layers = model_cfg['num_layers']
        
        self.node_embed = nn.Linear(node_dim, self.hidden_dim)
        self.edge_embed = nn.Linear(edge_dim, self.hidden_dim)
        
        self.convs = nn.ModuleList([
            ModernCGCNNLayer(
                node_dim=self.hidden_dim, 
                edge_dim=self.hidden_dim,
                use_batch_norm=model_cfg.get('batch_norm', True),
                use_residual=model_cfg.get('residual', True)
            ) for _ in range(num_layers)
        ])

    def encode(self, data) -> torch.Tensor:
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.node_embed(x)
        edge_attr = self.edge_embed(edge_attr)
        
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            
        x = global_mean_pool(x, batch)
        return x

class PredictionHead(nn.Module):
    """
    Dynamically configurable MLP for a specific prediction task.
    """
    def __init__(self, in_dim, out_dim=1, hidden_dim=None, dropout_rate=0.1, task_name="default"):
        super().__init__()
        self.task_name = task_name
        self.dropout_rate = dropout_rate
        if hidden_dim is None:
            hidden_dim = in_dim // 2
            
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # MC Dropout support: keep dropout active during inference if desired
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.silu(self.fc1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        out = self.fc2(x)
        return out.view(-1)

class TransferModel(nn.Module):
    """
    Combines a BaseCrystalEncoder with multiple PredictionHeads (Multi-Task Learning).
    Supports modular freezing strategies.
    """
    def __init__(self, encoder: BaseCrystalEncoder, heads: nn.ModuleDict):
        super().__init__()
        self.encoder = encoder
        self.heads = heads

    def forward(self, data):
        embedding = self.encoder.encode(data)
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(embedding)
        return outputs
        
    def freeze_strategy(self, strategy: str):
        """
        Applies a freezing strategy to the model.
        Available strategies:
        - "train_head_only": Freezes the entire encoder.
        - "partial_encoder": Freezes early layers of the encoder.
        - "full_finetuning": Unfreezes all layers.
        """
        logger.info(f"Applying freezing strategy: {strategy}")
        
        # Default: unfreeze all
        for param in self.parameters():
            param.requires_grad = True
            
        if strategy == "train_head_only":
            for param in self.encoder.parameters():
                param.requires_grad = False
                
        elif strategy == "partial_encoder":
            # For CGCNN, freeze node/edge embedding and first half of convs
            if hasattr(self.encoder, 'node_embed'):
                for param in self.encoder.node_embed.parameters(): param.requires_grad = False
                for param in self.encoder.edge_embed.parameters(): param.requires_grad = False
            
            if hasattr(self.encoder, 'convs'):
                n_convs = len(self.encoder.convs)
                for i in range(n_convs // 2):
                    for param in self.encoder.convs[i].parameters():
                        param.requires_grad = False
                        
        elif strategy == "full_finetuning":
            pass # Already handled by default unfreeze
        else:
            logger.warning(f"Unknown freezing strategy '{strategy}', defaulting to full_finetuning.")
