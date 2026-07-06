import os
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import StandardScaler
from .graph import structure_to_graph, GaussianDistance

class SuperconductorDataset(Dataset):
    """
    Dataset wrapper for processing and caching Superconductor graph data.
    """
    def __init__(self, structures, targets, config, scaler=None, root=None, is_train=True):
        super().__init__(root)
        self.structures = structures
        self.targets = targets
        self.config = config
        self.is_train = is_train
        
        self.rbf = GaussianDistance(
            dmin=config['data']['rbf_distance']['start'],
            dmax=config['data']['rbf_distance']['end'],
            step=config['data']['rbf_distance']['step']
        )
        
        # Precompute graphs
        self.graphs = []
        for s, t in zip(self.structures, self.targets):
            data = structure_to_graph(s, self.rbf, radius=config['data']['radius'], max_num_nbr=config['data']['max_num_nbr'])
            if isinstance(t, dict):
                # PyG DataLoader doesn't natively collate dicts well, so we set each target as an attribute
                for k, v in t.items():
                    setattr(data, f"y_{k}", torch.tensor([v], dtype=torch.float32))
            else:
                # Fallback for old pipeline single target
                data.y_default = torch.tensor([t], dtype=torch.float32)
                
            self.graphs.append(data)
            
        # Fit or apply scaler for node features to prevent data leakage
        all_x = torch.cat([g.x for g in self.graphs], dim=0).numpy()
        
        if scaler is None and self.is_train:
            self.scaler = StandardScaler()
            self.scaler.fit(all_x)
        else:
            self.scaler = scaler
            
        # Apply normalization
        for g in self.graphs:
            g.x = torch.tensor(self.scaler.transform(g.x.numpy()), dtype=torch.float32)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

def get_dataloaders(structures, targets, config):
    """
    Creates train, val, and test dataloaders with reproducible splits.
    """
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader
    
    val_split = config['data']['val_split']
    test_split = config['data']['test_split']
    seed = config['data']['random_seed']
    
    # Train / Val+Test
    s_train, s_temp, t_train, t_temp = train_test_split(
        structures, targets, test_size=(val_split + test_split), random_state=seed
    )
    
    # Val / Test
    rel_test_size = test_split / (val_split + test_split)
    s_val, s_test, t_val, t_test = train_test_split(
        s_temp, t_temp, test_size=rel_test_size, random_state=seed
    )
    
    train_dataset = SuperconductorDataset(s_train, t_train, config, is_train=True)
    val_dataset = SuperconductorDataset(s_val, t_val, config, scaler=train_dataset.scaler, is_train=False)
    test_dataset = SuperconductorDataset(s_test, t_test, config, scaler=train_dataset.scaler, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler
