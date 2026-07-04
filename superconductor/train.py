import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import os
import numpy as np
import json
import datetime
from superconductor.tasks import TaskRegistry
from superconductor.losses import get_loss_weighter

logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, loss_fns, loss_weighter, scaler, device, config):
    model.train()
    total_loss = 0
    clip_grad = config['training'].get('clip_grad_norm', 1.0)
    
    task_valid_samples = {t: 0 for t in loss_fns.keys()}
    task_total_samples = {t: 0 for t in loss_fns.keys()}
    
    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device.type, enabled=config['training'].get('mixed_precision', False)):
            out_dict = model(data)
            losses = {}
            for task_name, pred in out_dict.items():
                if hasattr(data, f'y_{task_name}'):
                    y = getattr(data, f'y_{task_name}').view(-1)
                elif hasattr(data, 'y_default'):
                    y = data.y_default.view(-1)
                else:
                    # Missing task completely from dataset, create NaN tensor
                    y = torch.full_like(pred.view(-1), float('nan'))
                
                # Mask out NaN values
                mask = ~torch.isnan(y)
                valid_count = mask.sum().item()
                task_valid_samples[task_name] += valid_count
                task_total_samples[task_name] += y.numel()
                
                if valid_count > 0:
                    raw_loss = loss_fns[task_name](pred, y)
                    # Use only valid losses
                    masked_loss = (raw_loss * mask).sum() / valid_count
                    losses[task_name] = masked_loss
                    
            batch_loss = loss_weighter(losses)
            
        if batch_loss > 0:
            scaler.scale(batch_loss).backward()
            
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += batch_loss.item() * data.num_graphs
            
    # Log sample stats
    for task_name in loss_fns.keys():
        valid = task_valid_samples[task_name]
        total = task_total_samples[task_name]
        if total > 0:
            logger.debug(f"{task_name}: {valid}/{total} valid labels ({(1 - valid/total)*100:.1f}% missing). Effective batch: {valid / len(dataloader):.1f}")
        
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, loss_fns, loss_weighter, device):
    model.eval()
    total_loss = 0
    
    all_preds = {t: [] for t in loss_fns.keys()}
    all_targets = {t: [] for t in loss_fns.keys()}
    
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            out_dict = model(data)
            
            losses = {}
            for task_name, pred in out_dict.items():
                if hasattr(data, f'y_{task_name}'):
                    y = getattr(data, f'y_{task_name}').view(-1)
                elif hasattr(data, 'y_default'):
                    y = data.y_default.view(-1)
                else:
                    y = torch.full_like(pred.view(-1), float('nan'))
                
                mask = ~torch.isnan(y)
                valid_count = mask.sum().item()
                
                if valid_count > 0:
                    raw_loss = loss_fns[task_name](pred, y)
                    losses[task_name] = (raw_loss * mask).sum() / valid_count
                
                # We save all predictions, but targets will have NaNs where missing
                all_preds[task_name].extend(pred.cpu().tolist())
                all_targets[task_name].extend(y.cpu().tolist())
                
            batch_loss = loss_weighter(losses)
            if isinstance(batch_loss, torch.Tensor):
                total_loss += batch_loss.item() * data.num_graphs
            
    return total_loss / len(dataloader.dataset), all_preds, all_targets

def train_model(model, train_loader, val_loader, config, save_dir="checkpoints", trial=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    task_registry = TaskRegistry(config)
    loss_fns = task_registry.get_loss_fns()
    loss_weighter = get_loss_weighter(config, task_registry.get_task_names(), task_registry.get_task_weights())
    loss_weighter = loss_weighter.to(device)
    
    params = list(model.parameters()) + list(loss_weighter.parameters())
    optimizer = torch.optim.AdamW(
        params, 
        lr=config['training']['learning_rate'],
        weight_decay=config['training'].get('weight_decay', 1e-5)
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.amp.GradScaler(device.type, enabled=config['training'].get('mixed_precision', False))
    
    epochs = config['training']['epochs']
    patience = config['training']['patience']
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pth")
    
    from superconductor.logger import LocalLogger
    exp_logger = LocalLogger(save_dir)
    exp_logger.log_params(config)
    
    logger.info(f"Starting Multi-Task training on {device} for tasks: {task_registry.get_task_names()}")
    
    import time
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fns, loss_weighter, scaler, device, config)
        val_loss, val_preds, val_targets = evaluate(model, val_loader, loss_fns, loss_weighter, device)
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr:.6f}")
        
        exp_logger.log_metrics({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr
        }, step=epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save Metadata
            metadata = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model_name": config['model'].get('name', 'TransferModel'),
                "tasks": task_registry.tasks,
                "val_loss": val_loss,
                "config_snapshot": config
            }
            with open(os.path.join(save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f, indent=4)
                
            # Modular saving for TransferModel
            if hasattr(model, 'encoder') and hasattr(model, 'heads'):
                torch.save(model.encoder.state_dict(), os.path.join(save_dir, "encoder_weights.pth"))
                for task_name, head in model.heads.items():
                    torch.save(head.state_dict(), os.path.join(save_dir, f"prediction_head_{task_name}.pth"))
            else:
                torch.save(model.state_dict(), best_model_path)
                
            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss_weighter': loss_weighter.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, os.path.join(save_dir, "training_state.pth"))
            
        else:
            patience_counter += 1
            
        if trial is not None:
            import optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                logger.info("Trial pruned by Optuna.")
                raise optuna.TrialPruned()
                
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break
            
    # Resilient Checkpoint Loading
    try:
        if hasattr(model, 'encoder') and hasattr(model, 'heads'):
            enc_path = os.path.join(save_dir, "encoder_weights.pth")
            if os.path.exists(enc_path):
                model.encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
            for task_name, head in model.heads.items():
                head_path = os.path.join(save_dir, f"prediction_head_{task_name}.pth")
                if os.path.exists(head_path):
                    head.load_state_dict(torch.load(head_path, map_location=device, weights_only=True))
        else:
            if os.path.exists(best_model_path):
                model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    except Exception as e:
        logger.error(f"Failed to load best weights, shape mismatch or missing checkpoint: {e}")
        
    train_time = time.time() - start_time
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if device.type == 'cuda' else 0.0
    
    exp_logger.log_hardware_stats({
        "train_time_s": train_time,
        "trainable_parameters": num_params,
        "peak_gpu_mem_mb": peak_mem
    })
    exp_logger.finish()
        
    return model, best_val_loss
