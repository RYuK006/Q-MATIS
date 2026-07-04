import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging
import os
import json

logger = logging.getLogger(__name__)

def calculate_metrics(targets, preds, task_name="default"):
    # Filter out missing (NaN) targets
    mask = ~np.isnan(targets)
    valid_targets = np.array(targets)[mask]
    valid_preds = np.array(preds)[mask]
    
    if len(valid_targets) == 0:
        logger.warning(f"No valid targets found for task {task_name}. Skipping metrics.")
        return {"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan}
        
    mae = mean_absolute_error(valid_targets, valid_preds)
    mse = mean_squared_error(valid_targets, valid_preds)
    rmse = np.sqrt(mse)
    # R2 requires at least 2 samples, handle gracefully
    if len(valid_targets) > 1:
        r2 = r2_score(valid_targets, valid_preds)
    else:
        r2 = np.nan
    
    metrics = {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2)
    }
    
    logger.info(f"--- Metrics for {task_name} (n={len(valid_targets)}) ---")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
        
    return metrics

def plot_predictions(targets, preds, save_path="predictions.png", task="Tc"):
    # Filter out missing (NaN) targets
    mask = ~np.isnan(targets)
    valid_targets = np.array(targets)[mask]
    valid_preds = np.array(preds)[mask]
    
    if len(valid_targets) == 0:
        return
        
    plt.figure(figsize=(8, 8))
    plt.scatter(valid_targets, valid_preds, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(min(valid_targets), min(valid_preds))
    max_val = max(max(valid_targets), max(valid_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel(f"True {task}")
    plt.ylabel(f"Predicted {task}")
    plt.title(f"Parity Plot: Predicted vs True {task} (n={len(valid_targets)})")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Saved parity plot to {save_path}")

def generate_multi_task_dashboard(true_vals_dict, preds_dict, exp_dir, all_preds=None):
    """Generates an aggregate metric report and parity plots for all tasks."""
    dashboard = {}
    
    for task_name in true_vals_dict.keys():
        targets = np.array(true_vals_dict[task_name])
        preds = np.array(preds_dict[task_name])
        
        metrics = calculate_metrics(targets, preds, task_name=task_name)
        
        # Uncertainty Calibration Metrics
        if all_preds is not None and task_name in all_preds:
            import scipy.stats
            # all_preds[task_name] shape: (ensemble_size, num_samples)
            task_all_preds = np.array(all_preds[task_name])
            if task_all_preds.ndim == 2 and task_all_preds.shape[0] > 1:
                mask = ~np.isnan(targets)
                valid_targets = targets[mask]
                
                # Mean and std dev for valid samples
                valid_all_preds = task_all_preds[:, mask]
                ensemble_mean = np.mean(valid_all_preds, axis=0)
                ensemble_std = np.std(valid_all_preds, axis=0)
                
                # 95% Confidence Interval (Z=1.96)
                z = 1.96
                lower_bound = ensemble_mean - z * ensemble_std
                upper_bound = ensemble_mean + z * ensemble_std
                
                # Prediction Interval Coverage Probability (PICP)
                coverage = (valid_targets >= lower_bound) & (valid_targets <= upper_bound)
                picp = np.mean(coverage)
                
                # Mean Prediction Interval Width (MPIW)
                mpiw = np.mean(upper_bound - lower_bound)
                
                metrics['PICP'] = float(picp)
                metrics['MPIW'] = float(mpiw)
                logger.info(f"[{task_name}] PICP: {picp:.4f} | MPIW: {mpiw:.4f}")
                
        dashboard[task_name] = metrics
        
        plot_path = os.path.join(exp_dir, f"parity_plot_{task_name}.png")
        plot_predictions(targets, preds, save_path=plot_path, task=task_name)
        
    # Save aggregate dashboard JSON
    dash_path = os.path.join(exp_dir, "metrics_dashboard.json")
    with open(dash_path, "w") as f:
        json.dump(dashboard, f, indent=4)
    logger.info(f"Saved aggregate metrics dashboard to {dash_path}")
    return dashboard
