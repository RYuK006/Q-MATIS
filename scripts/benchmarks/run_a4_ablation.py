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

def _train_and_eval(config, structures, targets, run_name, device):
    task_registry = TaskRegistry(config)
    node_dim = get_node_feature_dim()
    dmin = config['data']['rbf_distance']['start']
    dmax = config['data']['rbf_distance']['end']
    step = config['data']['rbf_distance']['step']
    edge_dim = int((dmax - dmin) / step) + 1
    
    config['model']['node_dim'] = node_dim
    config['model']['edge_dim'] = edge_dim
    
    train_loader, val_loader, test_loader, _ = get_dataloaders(structures, targets, config)
    
    encoder_name = config['model'].get('encoder_name', 'cgcnn')
    encoder = EncoderRegistry.build(encoder_name, config)
    heads = task_registry.build_heads(in_dim=encoder.hidden_dim)
    model = TransferModel(encoder, heads)
    
    start_time = time.time()
    trained_model, _ = train_model(model, train_loader, val_loader, config, save_dir=f"checkpoints/ablation_{run_name}")
    train_time = time.time() - start_time
    
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
        
    return results, train_time, inf_speed

def run_ablation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A4 Ablation on {device}")
    
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    base_config['pipeline_limits']['data_limit'] = 1000
    base_config['training']['epochs'] = 5
    
    # Force single worker for stability
    base_config['training']['batch_size'] = 32
    from superconductor.data_sources.build_dataset import DataOrchestrator
    orchestrator = DataOrchestrator(base_config)
    dataset = orchestrator.build_dataset(limit=base_config['pipeline_limits']['data_limit'])
    
    structures = [d['structure'] for d in dataset]
    targets = [{'tc': float(d.get('target', {}).get('tc', np.nan)), 'formation_energy': float(d.get('target', {}).get('formation_energy', np.nan))} for d in dataset]
    
    # 1. Single Task (Tc only)
    config_single = dict(base_config)
    config_single['tasks'] = [{'name': 'tc', 'target_key': 'tc', 'weight': 1.0}]
    print("\n--- Training Single Task (Tc) ---")
    res_single, t_single, i_single = _train_and_eval(config_single, structures, targets, "single", device)
    
    # 2. Multi-Task
    config_multi = dict(base_config)
    config_multi['tasks'] = [
        {'name': 'tc', 'target_key': 'tc', 'weight': 1.0},
        {'name': 'formation_energy', 'target_key': 'formation_energy', 'weight': 0.5}
    ]
    print("\n--- Training Multi Task (Tc + FE) ---")
    res_multi, t_multi, i_multi = _train_and_eval(config_multi, structures, targets, "multi", device)
    
    # Compare
    st_tc = res_single['tc']
    mt_tc = res_multi['tc']
    
    def pct_change(st, mt):
        if np.isnan(st) or np.isnan(mt) or st == 0: return 0.0
        return ((st - mt) / st) * 100
        
    report = f"""# Milestone A4 Multi-Task Ablation Results

## Tc Prediction Performance
| Metric | Single-Task | Multi-Task | Improvement |
|---|---|---|---|
| MAE | {st_tc['MAE']:.4f} | {mt_tc['MAE']:.4f} | {pct_change(st_tc['MAE'], mt_tc['MAE']):.2f}% |
| RMSE | {st_tc['RMSE']:.4f} | {mt_tc['RMSE']:.4f} | {pct_change(st_tc['RMSE'], mt_tc['RMSE']):.2f}% |
| R2 | {st_tc['R2']:.4f} | {mt_tc['R2']:.4f} | - |

## Training Dynamics
| Metric | Single-Task | Multi-Task |
|---|---|---|
| Train Time (s) | {t_single:.2f} | {t_multi:.2f} |
| Inference (samples/s) | {i_single:.2f} | {i_multi:.2f} |

"""
    print(report)
    with open("ablation_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    run_ablation()
