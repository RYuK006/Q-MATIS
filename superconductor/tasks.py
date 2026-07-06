import torch
import torch.nn as nn
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class TaskRegistry:
    """
    Registry for dynamic multi-task configuration.
    Parses configuration to create heads, loss functions, and metrics for each task.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tasks = []
        
        tasks_config = config.get('tasks', [])
        for t_cfg in tasks_config:
            if t_cfg.get('enabled', True):
                self.tasks.append(t_cfg)
                
        if not self.tasks:
            logger.warning("No tasks enabled in config! Defaulting to single 'tc' task.")
            self.tasks.append({
                'name': 'tc',
                'target_key': 'tc',
                'loss': 'huber',
                'metrics': ['mae', 'rmse', 'r2'],
                'weight': 1.0,
                'enabled': True
            })

    def get_task_names(self) -> List[str]:
        return [t['name'] for t in self.tasks]

    def get_target_keys(self) -> List[str]:
        return [t['target_key'] for t in self.tasks]

    def build_heads(self, in_dim: int, dropout_rate: float = 0.1) -> nn.ModuleDict:
        from superconductor.models import PredictionHead
        heads = nn.ModuleDict()
        for t in self.tasks:
            heads[t['name']] = PredictionHead(in_dim=in_dim, out_dim=1, dropout_rate=dropout_rate, task_name=t['name'])
        return heads

    def get_loss_fns(self) -> Dict[str, nn.Module]:
        loss_fns = {}
        for t in self.tasks:
            l_type = t.get('loss', 'mse').lower()
            if l_type == 'huber':
                loss_fns[t['name']] = nn.HuberLoss(reduction='none')
            elif l_type == 'mse':
                loss_fns[t['name']] = nn.MSELoss(reduction='none')
            elif l_type == 'l1' or l_type == 'mae':
                loss_fns[t['name']] = nn.L1Loss(reduction='none')
            else:
                logger.warning(f"Unknown loss type {l_type} for task {t['name']}. Using MSE.")
                loss_fns[t['name']] = nn.MSELoss(reduction='none')
        return loss_fns

    def get_task_weights(self) -> Dict[str, float]:
        return {t['name']: t.get('weight', 1.0) for t in self.tasks}

    def get_task_metrics(self) -> Dict[str, List[str]]:
        return {t['name']: t.get('metrics', ['mae']) for t in self.tasks}
