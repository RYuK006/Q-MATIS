from superconductor.compat import apply_platform_patches
apply_platform_patches()

import yaml
import logging
import torch
import os
import random
import numpy as np

from superconductor.data import get_dataloaders
from superconductor.train import train_model, evaluate
from superconductor.eval import calculate_metrics, plot_predictions
from superconductor.candidate_gen import generate_substitutions
from superconductor.active_learning import rank_candidates
from superconductor.dft import export_vasp
def set_seed(seed=42):
    """Sets the random seed for reproducible and deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class DiscoveryPipeline:
    def __init__(self, config_path="config.yaml"):
        import datetime
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("experiments", f"run_{self.run_id}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        self.config = self._load_config(config_path)
        self.validate_config(self.config)
        self._save_config_snapshot()
        
        self.logger = self._setup_logging()
        
        seed = self.config.get('system', {}).get('random_seed', 42)
        set_seed(seed)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._log_system_stats(seed)

    def validate_config(self, config):
        """Validates configuration schema and mutually exclusive options."""
        # Check required sections
        for section in ['model', 'training', 'data', 'tasks']:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")
                
        # Validate pretraining
        pretrain = config.get('pretrain', {})
        if pretrain.get('enabled', False):
            # Pretraining is now independent of dataset name, but we require a task
            if not any(t.get('target_key') == 'formation_energy' for t in config['tasks']):
                self.logger.warning("Pretraining is enabled but 'formation_energy' task is not in tasks config.")
                
        # Validate tasks
        tasks = config.get('tasks', [])
        if not tasks:
            raise ValueError("At least one task must be defined in config['tasks']")
            
        for t in tasks:
            if 'name' not in t or 'target_key' not in t:
                raise ValueError(f"Task must have 'name' and 'target_key'. Got: {t}")

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def _save_config_snapshot(self):
        snapshot_path = os.path.join(self.exp_dir, "config_snapshot.yaml")
        with open(snapshot_path, 'w') as f:
            yaml.dump(self.config, f)

    def _setup_logging(self):
        log_level_str = self.config.get('system', {}).get('log_level', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logger = logging.getLogger("DiscoveryPipeline")
        logger.setLevel(log_level)
        logger.propagate = False
        
        if logger.hasHandlers():
            logger.handlers.clear()
            
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        log_path = os.path.join(self.exp_dir, "pipeline.log")
        fh = logging.FileHandler(log_path)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def _log_system_stats(self, seed):
        import platform
        self.logger.info("="*50)
        self.logger.info(f"Experiment ID: {self.run_id}")
        self.logger.info(f"Output Directory: {self.exp_dir}")
        self.logger.info(f"Platform: {platform.system()} {platform.release()}")
        self.logger.info(f"Python Version: {platform.python_version()}")
        self.logger.info(f"PyTorch Version: {torch.__version__}")
        self.logger.info(f"Device: {self.device}")
        if self.device.type == 'cuda':
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        self.logger.info(f"Random Seed: {seed}")
        self.logger.info("="*50)

    def run(self):
        self.logger.info("Starting Discovery Pipeline Run")
        
        pretrain_config = self.config.get('pretrain', {})
        if pretrain_config.get('enabled', False):
            self.logger.info("=== STAGE 1: PRETRAINING ===")
            self._run_pretraining()
            self.logger.info("=== STAGE 2: FINE-TUNING ===")
        else:
            self.logger.info("Pretraining disabled. Starting direct training.")
            
        # Fine-tuning / Direct Training
        dataset = self._load_data()
        if not dataset:
            return
            
        structures, targets = self._prepare_structures(dataset)
        train_loader, val_loader, test_loader, scaler = get_dataloaders(structures, targets, self.config)
        
        if self.config.get('hpo', {}).get('enabled', False):
            self.logger.info("Starting Hyperparameter Optimization Phase")
            self._run_hpo(structures, targets)
            train_loader, val_loader, test_loader, scaler = get_dataloaders(structures, targets, self.config)
            
        # 3. Fine-tuning / Main Training
        self.logger.info("Starting Multi-Task Fine-Tuning...")
        models, true_vals, preds, all_preds = self._train_ensemble(train_loader, val_loader, test_loader)
        
        self._evaluate_ensemble(true_vals, preds, all_preds=all_preds)
        self._run_active_learning(models, dataset, scaler)

    def _run_pretraining(self):
        limit = self.config.get('pipeline_limits', {}).get('data_limit', 0)
        
        from superconductor.data_sources.mp import MPDataSource
        mp_ds = MPDataSource(
            api_key=self.config['data_sources'].get('api_key', ''),
            cache_dir=self.config['data_sources'].get('dataset_cache_dir', 'data/cache')
        )
        mp_data = mp_ds.fetch_data(limit=limit)
        
        if not mp_data:
            self.logger.warning("No pretraining data available. Skipping pretraining.")
            return
            
        structures = [d['structure'] for d in mp_data]
        targets = [d['target'] for d in mp_data] # Formation Energy
        
        # We can duplicate if toy dataset limit is too small for a valid split
        if len(structures) < 20:
            structures = structures * 10
            targets = targets * 10
            
        # Pretrain data is just one source, so we construct target dicts for it manually here
        from superconductor.tasks import TaskRegistry
        import numpy as np
        target_keys = TaskRegistry(self.config).get_target_keys()
        
        dict_targets = []
        for t in targets:
            task_targets = {}
            for key in target_keys:
                if key == 'formation_energy':
                    task_targets[key] = float(t)
                else:
                    task_targets[key] = np.nan
            dict_targets.append(task_targets)
            
        train_loader, val_loader, test_loader, _ = get_dataloaders(structures, dict_targets, self.config)
        
        # Temporarily use pretraining learning rate if specified
        original_lr = self.config['training']['learning_rate']
        self.config['training']['learning_rate'] = self.config['pretrain'].get('pretrain_lr', original_lr)
        
        # Train pretraining model (no ensemble needed for base pretraining)
        self.logger.info("Training BaseCrystalEncoder on MP Formation Energy...")
        models, true_vals, preds = self._train_ensemble(train_loader, val_loader, test_loader, ensemble_size=1, is_pretrain=True)
        
        self.logger.info("Evaluating Pretraining (Formation Energy) on Test Set...")
        self._evaluate_ensemble(true_vals, preds)
        
        # Restore LR
        self.config['training']['learning_rate'] = original_lr

    def _load_data(self):
        self.logger.info("Loading SuperCon dataset...")
        os.environ["MP_API_KEY"] = self.config['data_sources'].get('api_key', '')
        limit = self.config.get('pipeline_limits', {}).get('data_limit', 0)
        
        from superconductor.data_sources.build_dataset import build_dataset
        dataset = build_dataset(
            supercon_path=self.config['data_sources']['supercon_csv'],
            cache_dir=self.config['data_sources']['dataset_cache_dir'],
            limit=limit
        )
        if not dataset:
            self.logger.error("Dataset is empty.")
        return dataset

    def _prepare_structures(self, dataset):
        from superconductor.tasks import TaskRegistry
        import numpy as np
        
        task_registry = TaskRegistry(self.config)
        target_keys = task_registry.get_target_keys()
        
        structures = []
        targets = []
        
        for d in dataset:
            structures.append(d['structure'])
            raw_target = d.get('target', {})
            if not isinstance(raw_target, dict):
                # Legacy fallback if a data source hasn't been updated
                raw_target = {'tc': float(raw_target)}
                
            task_targets = {}
            for key in target_keys:
                task_targets[key] = float(raw_target.get(key, np.nan))
            targets.append(task_targets)
        
        if len(structures) < 20:
            self.logger.warning(f"Only {len(structures)} structures found. Duplicating to allow train/test split (toy mode).")
            structures = structures * 10
            targets = targets * 10
            
        self.logger.info(f"Prepared {len(structures)} structures for training.")
        return structures, targets

    def _run_hpo(self, structures, targets):
        from superconductor.hpo import run_hpo_study
        best_params = run_hpo_study(structures, targets, self.config)
        
        self.config['model']['hidden_dim'] = best_params['hidden_dim']
        self.config['model']['num_layers'] = best_params['num_layers']
        self.config['model']['dropout'] = best_params['dropout']
        self.config['training']['learning_rate'] = best_params['learning_rate']
        self.config['training']['weight_decay'] = best_params['weight_decay']
        self.config['training']['batch_size'] = best_params['batch_size']
        self.logger.info("Config updated with optimal HPO parameters.")

    def _train_ensemble(self, train_loader, val_loader, test_loader, ensemble_size=None, is_pretrain=False):
        if ensemble_size is None:
            ensemble_size = self.config.get('ensemble', {}).get('n_models', 1) if self.config.get('ensemble', {}).get('enabled', False) else 1
            
        self.logger.info(f"Training an ensemble of {ensemble_size} Multi-Task models...")
        
        from superconductor.models import EncoderRegistry, TransferModel
        from superconductor.features import get_node_feature_dim
        from superconductor.tasks import TaskRegistry
        from superconductor.train import train_model, evaluate, get_loss_weighter
        
        task_registry = TaskRegistry(self.config)
        loss_fns = task_registry.get_loss_fns()
        loss_weighter = get_loss_weighter(self.config, task_registry.get_task_names(), task_registry.get_task_weights())
        loss_weighter = loss_weighter.to(self.device)
        
        node_dim = get_node_feature_dim()
        dmin = self.config['data']['rbf_distance']['start']
        dmax = self.config['data']['rbf_distance']['end']
        step = self.config['data']['rbf_distance']['step']
        edge_dim = int((dmax - dmin) / step) + 1
        
        self.config['model']['node_dim'] = node_dim
        self.config['model']['edge_dim'] = edge_dim
        
        models = []
        all_preds = {t: [] for t in task_registry.get_task_names()}
        true_vals = None
        
        for i in range(ensemble_size):
            self.logger.info(f"--- Training Model {i+1}/{ensemble_size} ---")
            
            encoder_name = self.config['model'].get('encoder_name', 'cgcnn')
            encoder = EncoderRegistry.build(encoder_name, self.config)
            heads = task_registry.build_heads(in_dim=encoder.hidden_dim, dropout_rate=self.config['model'].get('dropout', 0.1))
            model = TransferModel(encoder, heads)
            
            # Load pretrained encoder weights if fine-tuning
            if not is_pretrain and self.config.get('pretrain', {}).get('enabled', False):
                pretrain_encoder_path = os.path.join(self.exp_dir, "model_0", "encoder_weights.pth")
                if os.path.exists(pretrain_encoder_path):
                    self.logger.info(f"Loading pretrained encoder weights from {pretrain_encoder_path}")
                    try:
                        model.encoder.load_state_dict(torch.load(pretrain_encoder_path, map_location=self.device, weights_only=True))
                        # Apply freezing strategy
                        strategy = self.config.get('pretrain', {}).get('freeze_strategy', 'full_finetuning')
                        model.freeze_strategy(strategy)
                    except Exception as e:
                        self.logger.error(f"Failed to load pretrained weights! Shape mismatch or corrupted file: {e}")
                else:
                    self.logger.warning("Pretrained encoder weights not found! Falling back to random init.")
            
            save_dir = os.path.join(self.exp_dir, f"model_{i}")
            trained_model, _ = train_model(model, train_loader, val_loader, self.config, save_dir=save_dir)
            models.append(trained_model)
            
            test_loss, preds_dict, true_vals_dict = evaluate(trained_model, test_loader, loss_fns, loss_weighter, self.device)
            self.logger.info(f"Model {i+1} Test Loss (Weighted): {test_loss:.4f}")
            
            for t_name in task_registry.get_task_names():
                all_preds[t_name].append(preds_dict[t_name])
                
            true_vals = true_vals_dict
            
        ensemble_preds = {}
        import numpy as np
        for t_name in task_registry.get_task_names():
            ensemble_preds[t_name] = np.mean(all_preds[t_name], axis=0)
            
        return models, true_vals, ensemble_preds, all_preds

    def _evaluate_ensemble(self, true_vals, ensemble_preds, all_preds=None):
        self.logger.info("Evaluating Ensemble Mean on Test Set...")
        from superconductor.eval import generate_multi_task_dashboard
        
        generate_multi_task_dashboard(true_vals, ensemble_preds, self.exp_dir, all_preds=all_preds)

    def _run_active_learning(self, models, dataset, scaler):
        self.logger.info("Starting Active Learning Discovery Loop...")
        base_structure = dataset[0]['structure'].copy()
        
        substitutions = [{'Mg': 'Y'}, {'Mg': 'Ba'}]
        candidates = generate_substitutions(base_structure, substitutions)
        
        ranked = rank_candidates(models, candidates, self.config, scaler, self.device)
        
        for i, res in enumerate(ranked):
            self.logger.info(f"Rank {i+1}: Tc={res['predicted_tc']:.2f}K | Uncertainty={res['uncertainty']:.2f} | Utility={res['utility']:.2f}")
            if res['predicted_tc'] >= self.config['active_learning']['tc_threshold']:
                out_dir = os.path.join(self.exp_dir, "dft_exports", f"candidate_{i}")
                export_vasp(res['structure'], out_dir, res['predicted_tc'])
                self.logger.info(f"  -> Exported to {out_dir}")
            else:
                self.logger.info("  -> Rejected: Below Tc threshold.")

if __name__ == "__main__":
    pipeline = DiscoveryPipeline()
    pipeline.run()
