import torch
import torch.nn as nn
from typing import Dict, Any

class BaseLossWeighter(nn.Module):
    def __init__(self, task_names):
        super().__init__()
        self.task_names = task_names

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

class ManualLossWeighter(BaseLossWeighter):
    def __init__(self, task_names, weights_dict):
        super().__init__(task_names)
        self.weights = weights_dict

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for task in self.task_names:
            if task in losses:
                total_loss += losses[task] * self.weights.get(task, 1.0)
        return total_loss

class UncertaintyLossWeighter(BaseLossWeighter):
    """
    Implements multi-task learning using homoscedastic uncertainty.
    Loss = sum(L_i / (2 * sigma_i^2) + log(sigma_i))
    where log(sigma_i^2) is learned to avoid division by zero.
    """
    def __init__(self, task_names):
        super().__init__(task_names)
        # Initialize log_vars to 0 (which means var=1.0)
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1)) for task in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for task in self.task_names:
            if task in losses:
                log_var = self.log_vars[task]
                precision = torch.exp(-log_var)
                loss = losses[task]
                total_loss += 0.5 * precision * loss + 0.5 * log_var
        return total_loss

def get_loss_weighter(config: Dict[str, Any], task_names, weights_dict=None):
    strategy = config.get('loss_weighting', {}).get('strategy', 'manual').lower()
    if strategy == 'uncertainty':
        return UncertaintyLossWeighter(task_names)
    else:
        # Default to manual weighting
        if weights_dict is None:
            weights_dict = {t: 1.0 for t in task_names}
        return ManualLossWeighter(task_names, weights_dict)
