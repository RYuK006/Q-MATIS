import os
from superconductor.compat import apply_platform_patches
apply_platform_patches()
import yaml
import time
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    trained_model, history = train_model(model, train_loader, val_loader, config, save_dir=f"checkpoints/benchmark_{encoder_name}")
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
        
    return results, train_time, inf_speed, peak_mem, num_params, history

def run_alignn_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running A5 Architecture Benchmark on {device}")
    
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    # We already resolved the dataset in a separate process!
    # Set api_key to None to prevent importing mp_api.client and crashing PyTorch
    if 'data_sources' in base_config:
        base_config['data_sources']['api_key'] = None
    
    base_config['pipeline_limits']['data_limit'] = 0
    base_config['training']['epochs'] = 50
    base_config['training']['batch_size'] = 32
    
    # We will benchmark on single task (Tc) for direct architecture comparison
    base_config['tasks'] = [{'name': 'tc', 'target_key': 'tc', 'weight': 1.0}]
    from superconductor.data_sources.build_dataset import DataOrchestrator
    
    seeds = [42, 123, 456]
    cgcnn_maes, cgcnn_rmses = [], []
    alignn_maes, alignn_rmses = [], []
    
    report_lines = [
        "# Milestone A5 Architecture Benchmark: CGCNN vs ALIGNN",
        "\n## Overview",
        "This benchmark evaluates ALIGNN against our baseline CGCNN.",
        "Both models were trained using 3 random seeds, 50 epochs, identical node/edge features, and identical early stopping constraints.",
        "\n## Loss Curves (Seed 42)"
    ]
    
    for idx, seed in enumerate(seeds):
        base_config['data']['random_seed'] = seed
        orchestrator = DataOrchestrator(base_config)
        dataset = orchestrator.build_dataset(limit=base_config['pipeline_limits']['data_limit'])
        
        structures = [d['structure'] for d in dataset]
        targets = [{'tc': float(d.get('target', {}).get('tc', np.nan))} for d in dataset]
        
        print(f"\n=== Seed {seed} ===")
        print("--- Training CGCNN Baseline ---")
        res_cgcnn, t_cgcnn, i_cgcnn, mem_cgcnn, param_cgcnn, hist_cgcnn = _train_and_eval(base_config, structures, targets, "cgcnn", device)
        cgcnn_maes.append(res_cgcnn['tc']['MAE'])
        cgcnn_rmses.append(res_cgcnn['tc']['RMSE'])
        
        print("--- Training ALIGNN ---")
        try:
            res_alignn, t_alignn, i_alignn, mem_alignn, param_alignn, hist_alignn = _train_and_eval(base_config, structures, targets, "alignn", device)
            alignn_maes.append(res_alignn['tc']['MAE'])
            alignn_rmses.append(res_alignn['tc']['RMSE'])
            alignn_success = True
        except Exception as e:
            print(f"ALIGNN Benchmark failed: {e}")
            alignn_success = False
            
        if idx == 0 and alignn_success:
            report_lines.append("\n### CGCNN and ALIGNN (Seed 42) Loss curves are saved as images in checkpoints/.")
    
    if alignn_success:
        cgcnn_mae_mean, cgcnn_mae_std = np.mean(cgcnn_maes), np.std(cgcnn_maes)
        cgcnn_rmse_mean, cgcnn_rmse_std = np.mean(cgcnn_rmses), np.std(cgcnn_rmses)
        alignn_mae_mean, alignn_mae_std = np.mean(alignn_maes), np.std(alignn_maes)
        alignn_rmse_mean, alignn_rmse_std = np.mean(alignn_rmses), np.std(alignn_rmses)
        
        report_lines.append("\n## Tc Prediction Performance (3 Seeds, 50 Epochs)")
        report_lines.append("| Metric | CGCNN (Mean ± Std) | ALIGNN (Mean ± Std) |")
        report_lines.append("|---|---|---|")
        report_lines.append(f"| MAE | {cgcnn_mae_mean:.4f} ± {cgcnn_mae_std:.4f} | {alignn_mae_mean:.4f} ± {alignn_mae_std:.4f} |")
        report_lines.append(f"| RMSE | {cgcnn_rmse_mean:.4f} ± {cgcnn_rmse_std:.4f} | {alignn_rmse_mean:.4f} ± {alignn_rmse_std:.4f} |")
        
        report = "\n".join(report_lines)
        print("\n\n" + report)
        with open("alignn_benchmark_report.md", "w") as f:
            f.write(report)

if __name__ == "__main__":
    try:
        run_alignn_benchmark()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
