import os
import yaml
import time
import torch
import numpy as np

from superconductor.data import get_dataloaders
from superconductor.models import EncoderRegistry, TransferModel
from superconductor.tasks import TaskRegistry
from superconductor.train import train_model, evaluate, get_loss_weighter
from superconductor.data_sources.build_dataset import build_dataset
from superconductor.features import get_node_feature_dim

def calculate_model_stats(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params

def _train_and_eval(config, structures, targets, encoder_name, device):
    task_registry = TaskRegistry(config)
    node_dim = get_node_feature_dim()
    dmin = config['data']['rbf_distance']['start']
    dmax = config['data']['rbf_distance']['end']
    step = config['data']['rbf_distance']['step']
    edge_dim = int((dmax - dmin) / step) + 1
    
    config['model']['node_dim'] = node_dim
    config['model']['edge_dim'] = edge_dim
    config['model']['encoder_name'] = encoder_name
    
    train_loader, val_loader, test_loader, _ = get_dataloaders(structures, targets, config)
    
    encoder = EncoderRegistry.build(encoder_name, config)
    heads = task_registry.build_heads(in_dim=encoder.hidden_dim)
    model = TransferModel(encoder, heads)
    
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    
    start_time = time.time()
    trained_model, _ = train_model(model, train_loader, val_loader, config, save_dir=f"checkpoints/benchmark_{encoder_name}")
    train_time = time.time() - start_time
    
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    num_params = calculate_model_stats(model)
    
    inf_start = time.time()
    loss_fns = task_registry.get_loss_fns()
    loss_weighter = get_loss_weighter(config, task_registry.get_task_names(), task_registry.get_task_weights()).to(device)
    
    _, preds_dict, true_dict = evaluate(trained_model, test_loader, loss_fns, loss_weighter, device)
    inf_time = time.time() - inf_start
    inf_speed = len(test_loader.dataset) / inf_time
    
    from superconductor.eval import calculate_metrics
    results = {}
    for t_name in task_registry.get_task_names():
        m = calculate_metrics(np.array(true_dict[t_name]), np.array(preds_dict[t_name]), task_name=t_name)
        results[t_name] = m
        
    return results, train_time, inf_speed, peak_mem, num_params

def run_alignn_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A5 Architecture Benchmark on {device}")
    
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    base_config['pipeline_limits']['data_limit'] = 1000
    base_config['training']['epochs'] = 5
    base_config['training']['batch_size'] = 32
    
    # We will benchmark on single task (Tc) for direct architecture comparison
    base_config['tasks'] = [{'name': 'tc', 'target_key': 'tc', 'weight': 1.0}]
    from superconductor.data_sources.build_dataset import DataOrchestrator
    orchestrator = DataOrchestrator(base_config)
    dataset = orchestrator.build_dataset(limit=base_config['pipeline_limits']['data_limit'])
    
    structures = [d['structure'] for d in dataset]
    targets = [{'tc': float(d.get('target', {}).get('tc', np.nan))} for d in dataset]
    
    print("\n--- Training CGCNN Baseline ---")
    res_cgcnn, t_cgcnn, i_cgcnn, mem_cgcnn, param_cgcnn = _train_and_eval(base_config, structures, targets, "cgcnn", device)
    
    print("\n--- Training ALIGNN ---")
    try:
        res_alignn, t_alignn, i_alignn, mem_alignn, param_alignn = _train_and_eval(base_config, structures, targets, "alignn", device)
        alignn_success = True
    except Exception as e:
        print(f"ALIGNN Benchmark failed (likely due to missing DGL or mock fallback): {e}")
        alignn_success = False
    
    if alignn_success:
        report = f"""# Milestone A5 Architecture Benchmark: CGCNN vs ALIGNN

## Overview
This benchmark evaluates the state-of-the-art ALIGNN (Atomistic Line Graph Neural Network) against our baseline CGCNN.
Both models were trained using exactly the same data split (seed={base_config['data']['random_seed']}), identical node/edge features, and identical early stopping constraints.

## Tc Prediction Performance
| Metric | CGCNN | ALIGNN | Improvement |
|---|---|---|---|
| MAE | {res_cgcnn['tc']['MAE']:.4f} | {res_alignn['tc']['MAE']:.4f} | {((res_cgcnn['tc']['MAE'] - res_alignn['tc']['MAE']) / res_cgcnn['tc']['MAE']) * 100:.2f}% |
| RMSE | {res_cgcnn['tc']['RMSE']:.4f} | {res_alignn['tc']['RMSE']:.4f} | {((res_cgcnn['tc']['RMSE'] - res_alignn['tc']['RMSE']) / res_cgcnn['tc']['RMSE']) * 100:.2f}% |
| R2 | {res_cgcnn['tc']['R2']:.4f} | {res_alignn['tc']['R2']:.4f} | - |

## Computational Profile
| Metric | CGCNN | ALIGNN |
|---|---|---|
| Train Time (s) | {t_cgcnn:.2f} | {t_alignn:.2f} |
| Inference (samples/s) | {i_cgcnn:.2f} | {i_alignn:.2f} |
| Peak GPU Mem (MB) | {mem_cgcnn:.2f} | {mem_alignn:.2f} |
| Parameters | {param_cgcnn:,} | {param_alignn:,} |

"""
        print(report)
        with open("alignn_benchmark_report.md", "w") as f:
            f.write(report)

if __name__ == "__main__":
    run_alignn_benchmark()
