import torch
import numpy as np
import logging
from .graph import structure_to_graph

logger = logging.getLogger(__name__)

def estimate_uncertainty(models, graph_data, device):
    """
    Uses Deep Ensembles to estimate prediction uncertainty.
    """
    graph_data = graph_data.to(device)
    
    preds = []
    with torch.no_grad():
        for model in models:
            model.to(device)
            model.eval()
            out = model(graph_data)
            preds.append(out.item())
            
    preds = np.array(preds)
    mean_tc = np.mean(preds)
    uncertainty = np.std(preds)
    
    # Confidence score (inverse of relative uncertainty)
    confidence = 1.0 / (1.0 + uncertainty / (np.abs(mean_tc) + 1e-5))
    
    return mean_tc, uncertainty, confidence

def rank_candidates(models, structures, config, dataset_scaler, device):
    """
    Evaluates a list of candidate structures and ranks them by expected utility 
    (high Tc and high uncertainty for exploration).
    """
    from .data import SuperconductorDataset
    
    results = []
    
    # Dummy target since we are predicting
    dummy_targets = [0.0] * len(structures)
    dataset = SuperconductorDataset(structures, dummy_targets, config, scaler=dataset_scaler, is_train=False)
    
    for i, data in enumerate(dataset):
        data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        mean_tc, uncert, conf = estimate_uncertainty(models, data, device)
        
        results.append({
            'index': i,
            'structure': structures[i],
            'predicted_tc': mean_tc,
            'uncertainty': uncert,
            'confidence': conf,
            # Utility function for acquisition (Upper Confidence Bound)
            'utility': mean_tc + 1.0 * uncert 
        })
        
    results.sort(key=lambda x: x['utility'], reverse=True)
    return results
