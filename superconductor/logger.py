from abc import ABC, abstractmethod
import os
import json
import csv
import logging
from datetime import datetime
import matplotlib.pyplot as plt

class BaseExperimentLogger(ABC):
    """
    Abstract interface for experiment tracking.
    Allows seamlessly swapping between local CSVs, MLflow, or Weights & Biases.
    """
    
    @abstractmethod
    def log_metrics(self, metrics: dict, step: int = None):
        """Log a dictionary of metrics."""
        pass
        
    @abstractmethod
    def log_params(self, params: dict):
        """Log a dictionary of hyperparameters or config values."""
        pass
        
    @abstractmethod
    def save_artifact(self, file_path: str):
        """Save a local file as an artifact."""
        pass
        
    @abstractmethod
    def log_hardware_stats(self, stats: dict):
        """Log hardware diagnostics (VRAM, throughput, param count)."""
        pass
        
    @abstractmethod
    def finish(self):
        """Close logger connections or finalize plotting."""
        pass

class LocalLogger(BaseExperimentLogger):
    """Default logger that saves to local JSON/CSV files."""
    
    def __init__(self, exp_dir: str):
        self.exp_dir = exp_dir
        self.logger = logging.getLogger(__name__)
        
        self.metrics_history = []
        self.params = {}
        self.hardware_stats = {}
        
        os.makedirs(self.exp_dir, exist_ok=True)
        self.csv_path = os.path.join(self.exp_dir, "training_log.csv")
        
    def log_metrics(self, metrics: dict, step: int = None):
        if step is not None:
            metrics['step'] = step
        else:
            metrics['step'] = len(self.metrics_history)
            
        self.metrics_history.append(metrics)
        
        # Append to CSV
        file_exists = os.path.isfile(self.csv_path)
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
            
    def log_params(self, params: dict):
        self.params.update(params)
        with open(os.path.join(self.exp_dir, "config_snapshot.json"), 'w') as f:
            json.dump(self.params, f, indent=4)
            
    def save_artifact(self, file_path: str):
        # Local logger assumes the file is already saved in the correct directory if it's local
        self.logger.info(f"Artifact saved: {file_path}")
        
    def log_hardware_stats(self, stats: dict):
        self.hardware_stats.update(stats)
        with open(os.path.join(self.exp_dir, "hardware_stats.json"), 'w') as f:
            json.dump(self.hardware_stats, f, indent=4)
            
    def finish(self):
        """Generates learning curves from tracked metrics."""
        if not self.metrics_history:
            return
            
        epochs = [m['step'] for m in self.metrics_history]
        
        plt.figure(figsize=(10, 6))
        
        # Find all keys that end with 'loss' or 'MAE'
        metric_keys = [k for k in self.metrics_history[0].keys() if k != 'step']
        
        for key in metric_keys:
            if 'loss' in key.lower() or 'mae' in key.lower():
                values = [m.get(key, None) for m in self.metrics_history]
                if all(v is not None for v in values):
                    plt.plot(epochs, values, label=key)
                    
        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.exp_dir, "learning_curves.png")
        plt.savefig(plot_path)
        plt.close()
        self.logger.info(f"Saved learning curves to {plot_path}")
