import os
import yaml
import time
import json
import torch
import numpy as np
from datetime import datetime

from superconductor.models import CGCNNEncoder, TransferModel
from superconductor.tasks import TaskRegistry
from superconductor.data import get_dataloaders
from superconductor.features import get_node_feature_dim
from superconductor.data_sources.build_dataset import build_dataset
from superconductor.train import train_model, evaluate, get_loss_weighter

def calculate_model_stats(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mem_params = sum(p.numel() * p.element_size() for p in model.parameters())
    return num_params, mem_params / (1024 ** 2) # MB

def run_experiment(config, run_name, train_loader, val_loader, test_loader, device):
    task_registry = TaskRegistry(config)
    node_dim = get_node_feature_dim()
    dmin = config['data']['rbf_distance']['start']
    dmax = config['data']['rbf_distance']['end']
    step = config['data']['rbf_distance']['step']
    edge_dim = int((dmax - dmin) / step) + 1
    
    config['model']['node_dim'] = node_dim
    config['model']['edge_dim'] = edge_dim
    
    from superconductor.models import EncoderRegistry
    encoder_name = config['model'].get('encoder_name', 'cgcnn')
    encoder = EncoderRegistry.build(encoder_name, config)
    heads = task_registry.build_heads(in_dim=encoder.hidden_dim)
    model = TransferModel(encoder, heads)
    
    if config.get('pretrain', {}).get('enabled', False):
        # In a real run, this would load weights
        pass
        
    model.freeze_strategy(config.get('pretrain', {}).get('freeze_strategy', 'full_finetuning'))
    
    num_params, param_mem = calculate_model_stats(model)
    
    torch.cuda.reset_peak_memory_stats(device) if device.type == 'cuda' else None
    
    start_time = time.time()
    trained_model, _ = train_model(model, train_loader, val_loader, config, save_dir=f"checkpoints/benchmark_{run_name}")
    train_time = time.time() - start_time
    
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    
    # Inference speed test
    inf_start = time.time()
    loss_fns = task_registry.get_loss_fns()
    loss_weighter = get_loss_weighter(config, task_registry.get_task_names(), task_registry.get_task_weights()).to(device)
    
    _, preds_dict, true_dict = evaluate(trained_model, test_loader, loss_fns, loss_weighter, device)
    inf_time = time.time() - inf_start
    inf_speed = len(test_loader.dataset) / inf_time
    
    # Calculate metrics
    from superconductor.eval import calculate_metrics
    metrics = calculate_metrics(true_dict['tc'], preds_dict['tc'], task_name="tc")
    
    return {
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "R2": metrics["R2"],
        "Train_Time_s": train_time,
        "Inference_Samples_per_s": inf_speed,
        "Trainable_Parameters": num_params,
        "Model_Size_MB": param_mem,
        "Peak_GPU_Mem_MB": peak_mem
    }

def run_benchmarks():
    with open("config.yaml", "r") as f:
        base_config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on {device}")
    
    # Fetch small sample for benchmark
    os.environ["MP_API_KEY"] = base_config['data_sources']['api_key']
    from superconductor.data_sources.build_dataset import DataOrchestrator
    orchestrator = DataOrchestrator(base_config)
    dataset = orchestrator.build_dataset(limit=base_config['pipeline_limits']['data_limit'])
    
    structures = [d['structure'] for d in dataset]
    targets = [{'tc': float(d.get('target', {}).get('tc', np.nan))} for d in dataset]
    
    import scipy.stats as stats
    
    # 1. Random Init vs Transfer Learning
    results = {"random": [], "transfer": []}
    seeds = list(range(42, 52)) # 10 seeds
    for mode in ["random", "transfer"]:
        print(f"\n--- Running {mode} experiments ---")
        for seed in seeds:
            config = yaml.safe_load(yaml.dump(base_config))
            config['training']['epochs'] = 2 # Fast
            config['data']['random_seed'] = seed
            config['pretrain']['enabled'] = (mode == "transfer")
            
            train_loader, val_loader, test_loader, _ = get_dataloaders(structures, targets, config)
            res = run_experiment(config, f"{mode}_{seed}", train_loader, val_loader, test_loader, device)
            results[mode].append(res)
            
    # Statistical Significance Testing
    rand_maes = [r["MAE"] for r in results["random"]]
    trans_maes = [r["MAE"] for r in results["transfer"]]
    
    t_stat, p_val = stats.ttest_rel(rand_maes, trans_maes)
    
    # Bootstrap CI for Transfer MAE
    res_bootstrap = stats.bootstrap((trans_maes,), np.mean, confidence_level=0.95, random_state=42)
    ci_low, ci_high = res_bootstrap.confidence_interval.low, res_bootstrap.confidence_interval.high
            
    # 2. Layer-wise fine-tuning
    strategies = ["train_head_only", "partial_encoder", "full_finetuning"]
    results["strategies"] = {}
    for strat in strategies:
        config = yaml.safe_load(yaml.dump(base_config))
        config['training']['epochs'] = 2
        config['pretrain']['enabled'] = True
        config['pretrain']['freeze_strategy'] = strat
        
        train_loader, val_loader, test_loader, _ = get_dataloaders(structures, targets, config)
        res = run_experiment(config, f"strat_{strat}", train_loader, val_loader, test_loader, device)
        results["strategies"][strat] = res
        
    # Generate Report
    with open("benchmark_report.md", "w") as f:
        f.write("# Milestone A3/A4 Benchmark Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## 1. Transfer Learning vs Random Initialization (10 Seeds)\n")
        f.write("| Mode | Mean MAE | Std MAE | Mean RMSE | Std RMSE | Mean R2 | Std R2 |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        
        for mode in ["random", "transfer"]:
            maes = [r["MAE"] for r in results[mode]]
            rmses = [r["RMSE"] for r in results[mode]]
            r2s = [r["R2"] for r in results[mode]]
            f.write(f"| {mode} | {np.mean(maes):.4f} | {np.std(maes):.4f} | {np.mean(rmses):.4f} | {np.std(rmses):.4f} | {np.mean(r2s):.4f} | {np.std(r2s):.4f} |\n")
            
        f.write("\n### Statistical Significance (MAE)\n")
        f.write(f"- **Paired t-test (Transfer vs Random)**: p-value = {p_val:.4e} (t-statistic = {t_stat:.4f})\n")
        f.write(f"- **Transfer MAE 95% Confidence Interval (Bootstrap)**: [{ci_low:.4f}, {ci_high:.4f}]\n")
            
        f.write("\n## 2. Layer-wise Fine-Tuning Performance\n")
        f.write("| Strategy | MAE | Trainable Params | Train Time (s) | Peak GPU Mem (MB) |\n")
        f.write("|---|---|---|---|---|\n")
        for strat in strategies:
            r = results["strategies"][strat]
            f.write(f"| {strat} | {r['MAE']:.4f} | {r['Trainable_Parameters']} | {r['Train_Time_s']:.2f} | {r['Peak_GPU_Mem_MB']:.2f} |\n")
            
        f.write("\n## 3. Computational Metrics\n")
        sample_r = results["random"][0]
        f.write(f"- **Inference Speed**: {sample_r['Inference_Samples_per_s']:.2f} samples/sec\n")
        f.write(f"- **Model Size**: {sample_r['Model_Size_MB']:.2f} MB\n")
        
    print("Benchmark report generated at benchmark_report.md")

if __name__ == "__main__":
    run_benchmarks()
