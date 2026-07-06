import optuna
import copy
import logging
from .train import train_model
from .data import get_dataloaders

logger = logging.getLogger(__name__)

def run_hpo_study(structures, targets, config):
    logger.info("Starting Hyperparameter Optimization with Optuna...")
    n_trials = config.get('hpo', {}).get('n_trials', 10)
    
    def objective(trial):
        # Deep copy config so we don't mutate the original
        trial_config = copy.deepcopy(config)
        
        # Sample hyperparameters
        trial_config['model']['hidden_dim'] = trial.suggest_categorical('hidden_dim', [64, 128, 256])
        trial_config['model']['num_layers'] = trial.suggest_int('num_layers', 2, 5)
        trial_config['model']['dropout'] = trial.suggest_float('dropout', 0.0, 0.5)
        trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        trial_config['training']['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        trial_config['training']['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        train_loader, val_loader, test_loader, scaler = get_dataloaders(structures, targets, trial_config)
        
        from .models import CGCNNEncoder, TransferModel
        from .tasks import TaskRegistry
        from .features import get_node_feature_dim
        
        task_registry = TaskRegistry(trial_config)
        trial_config['model']['node_dim'] = get_node_feature_dim()
        dmin = trial_config['data']['rbf_distance']['start']
        dmax = trial_config['data']['rbf_distance']['end']
        step = trial_config['data']['rbf_distance']['step']
        trial_config['model']['edge_dim'] = int((dmax - dmin) / step) + 1
        
        encoder = CGCNNEncoder(trial_config)
        heads = task_registry.build_heads(in_dim=encoder.hidden_dim, dropout_rate=trial_config['model'].get('dropout', 0.1))
        model = TransferModel(encoder, heads)
        
        # We don't want to save models for every trial to the main directory
        save_dir = f"checkpoints/trial_{trial.number}"
        try:
            _, best_val_loss = train_model(model, train_loader, val_loader, trial_config, save_dir=save_dir, trial=trial)
        except optuna.TrialPruned:
            raise
            
        return best_val_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("HPO Study complete.")
    logger.info(f"Best Trial: {study.best_trial.number}")
    logger.info(f"Best Value (Val Loss): {study.best_trial.value}")
    logger.info("Best Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")
        
    return study.best_trial.params
