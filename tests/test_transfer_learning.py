import torch
import pytest
from superconductor.models import CGCNNEncoder, PredictionHead, TransferModel

def test_freezing_strategies():
    config = {
        'model': {
            'node_dim': 17,
            'edge_dim': 40,
            'hidden_dim': 64,
            'num_layers': 4,
            'batch_norm': True,
            'residual': True,
            'dropout': 0.1
        }
    }
    
    encoder = CGCNNEncoder(config)
    head = PredictionHead(in_dim=encoder.hidden_dim, out_dim=1, task_name="tc")
    model = TransferModel(encoder, head)
    
    # Strategy: train_head_only
    model.freeze_strategy("train_head_only")
    for param in encoder.parameters():
        assert param.requires_grad == False
    for param in head.parameters():
        assert param.requires_grad == True
        
    # Strategy: full_finetuning
    model.freeze_strategy("full_finetuning")
    for param in encoder.parameters():
        assert param.requires_grad == True
    for param in head.parameters():
        assert param.requires_grad == True
        
    # Strategy: partial_encoder (freezes embeds + first half of convs)
    model.freeze_strategy("partial_encoder")
    for param in encoder.node_embed.parameters():
        assert param.requires_grad == False
    for param in encoder.edge_embed.parameters():
        assert param.requires_grad == False
    
    # Convs: 4 total. First 2 should be frozen.
    for i in range(2):
        for param in encoder.convs[i].parameters():
            assert param.requires_grad == False
    for i in range(2, 4):
        for param in encoder.convs[i].parameters():
            assert param.requires_grad == True
