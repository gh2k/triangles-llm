import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import Dict, Optional

from .models import TriangleGenerator
from .renderer import create_renderer
from .loss import CombinedLoss
from .datasets import create_dataloaders
from .utils import save_image, create_comparison_grid, MetricsCalculator


def get_warmup_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Create a learning rate scheduler with linear warmup and cosine decay."""
    def lr_lambda(step):
        if warmup_steps <= 0:
            # No warmup
            if total_steps <= 1:
                return 1.0  # Constant LR for very small datasets
            progress = step / total_steps
            return 0.5 * (1 + np.cos(np.pi * progress))
        elif step < warmup_steps:
            return step / warmup_steps
        else:
            if total_steps <= warmup_steps:
                return 1.0  # Constant LR if total steps <= warmup steps
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, step: int, loss: float, cfg: DictConfig, 
                   checkpoint_path: str) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'loss': loss,
        'config': OmegaConf.to_container(cfg),
        'triangles_n': cfg.triangles_n,
        'image_size': cfg.image_size
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def train_epoch(model: nn.Module, renderer, loss_fn: nn.Module,
               train_loader: DataLoader, optimizer: torch.optim.Optimizer,
               scheduler, scaler: GradScaler, epoch: int, cfg: DictConfig,
               metrics_calc: MetricsCalculator) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    # Initialize epoch losses - include both standard and geometric mean metrics
    epoch_losses = {
        'total': [], 'perceptual': [], 'l1': [], 'lpips': [],
        'geometric_mean': [], 'regularization': [],
        'log_perceptual': [], 'log_l1': [], 'log_lpips': []
    }
    epoch_metrics = {'lpips': [], 'ssim': [], 'psnr': []}
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(cfg.training.device)
        
        # Resize images to match renderer size if needed
        if images.shape[2] != cfg.image_size or images.shape[3] != cfg.image_size:
            images = F.interpolate(images, size=(cfg.image_size, cfg.image_size), mode='bilinear', align_corners=False)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(device_type='cuda', enabled=cfg.training.mixed_precision):
            # Generate triangle parameters
            triangle_params = model(images)
            
            # Split into coordinates and colors
            coordinates, colors = model.get_triangle_params(triangle_params)
            
            # DiffVG requires float32 for stable gradients
            # Exit autocast context for rendering
            with autocast(device_type='cuda', enabled=False):
                # Ensure coordinates and colors are float32
                coordinates = coordinates.float()
                colors = colors.float()
                
                # Render triangles
                rendered = renderer.render_batch(coordinates, colors)
            
            # Compute loss (back in autocast context)
            losses = loss_fn(rendered, images)
        
        # Backward pass
        scaler.scale(losses['total']).backward()
        
        # Check for NaN gradients before clipping
        has_nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                print(f"WARNING: NaN gradient detected in {name}")
                break
        
        if has_nan_grad:
            print("Skipping optimizer step due to NaN gradients")
            optimizer.zero_grad()
            scaler.update()
            continue
        
        # Gradient clipping
        if cfg.training.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        
        # Optimizer step
        scaler.step(optimizer)
        scaler.update()
        
        # Scheduler step
        scheduler.step()
        
        # Record losses (only if key exists in epoch_losses)
        for k, v in losses.items():
            if k in epoch_losses:
                epoch_losses[k].append(v.item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Log to W&B
        global_step = epoch * len(train_loader) + batch_idx
        
        if batch_idx % cfg.logging.log_interval == 0:
            log_dict = {
                'train/loss_total': losses['total'].item(),
                'train/loss_perceptual': losses['perceptual'].item(),
                'train/loss_l1': losses['l1'].item(),
                'train/loss_lpips': losses['lpips'].item(),
                'train/lr': scheduler.get_last_lr()[0],
                'train/epoch': epoch,
                'train/step': global_step
            }
            
            # Add geometric mean specific metrics if available
            if 'geometric_mean' in losses:
                log_dict['train/geometric_mean'] = losses['geometric_mean'].item()
            if 'regularization' in losses:
                log_dict['train/regularization'] = losses['regularization'].item()
            if 'log_perceptual' in losses:
                log_dict['train/log_perceptual'] = losses['log_perceptual'].item()
                log_dict['train/log_l1'] = losses['log_l1'].item()
                log_dict['train/log_lpips'] = losses['log_lpips'].item()
            
            # Calculate metrics periodically
            if batch_idx % (cfg.logging.log_interval * 5) == 0:
                with torch.no_grad():
                    metrics = metrics_calc.calculate_metrics(rendered, images)
                    for k, v in metrics.items():
                        log_dict[f'train/{k}'] = v
                        epoch_metrics[k].append(v)
            
            wandb.log(log_dict, step=global_step)
    
    # Return epoch averages (handle empty lists)
    avg_losses = {k: np.mean(v) if v else 0 for k, v in epoch_losses.items()}
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in epoch_metrics.items()}
    
    return {**avg_losses, **avg_metrics}


def validate(model: nn.Module, renderer, loss_fn: nn.Module,
            val_loader: DataLoader, cfg: DictConfig,
            metrics_calc: MetricsCalculator) -> Dict[str, float]:
    """Run validation."""
    model.eval()
    
    # Initialize validation losses - include both standard and geometric mean metrics
    val_losses = {
        'total': [], 'perceptual': [], 'l1': [], 'lpips': [],
        'geometric_mean': [], 'regularization': [],
        'log_perceptual': [], 'log_l1': [], 'log_lpips': []
    }
    val_metrics = {'lpips': [], 'ssim': [], 'psnr': []}
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            images = batch['image'].to(cfg.training.device)
            
            # Resize images to match renderer size if needed
            if images.shape[2] != cfg.image_size or images.shape[3] != cfg.image_size:
                images = F.interpolate(images, size=(cfg.image_size, cfg.image_size), mode='bilinear', align_corners=False)
            
            # Generate and render
            triangle_params = model(images)
            coordinates, colors = model.get_triangle_params(triangle_params)
            rendered = renderer.render_batch(coordinates, colors)
            
            # Compute losses
            losses = loss_fn(rendered, images)
            for k, v in losses.items():
                if k in val_losses:
                    val_losses[k].append(v.item())
            
            # Compute metrics
            metrics = metrics_calc.calculate_metrics(rendered, images)
            for k, v in metrics.items():
                val_metrics[k].append(v)
    
    # Return averages (handle empty lists)
    avg_losses = {k: np.mean(v) if v else 0 for k, v in val_losses.items()}
    avg_metrics = {k: np.mean(v) if v else 0 for k, v in val_metrics.items()}
    
    return {**avg_losses, **avg_metrics}


def train(cfg: DictConfig) -> None:
    """Main training function."""
    # Initialize W&B
    wandb.init(
        project=cfg.logging.wandb_project,
        entity=cfg.logging.wandb_entity,
        config=OmegaConf.to_container(cfg),
        name=f"triangulate_{cfg.triangles_n}_{cfg.image_size}"
    )
    
    # Set device
    device = torch.device(cfg.training.device)
    
    # Create model
    model = TriangleGenerator(
        triangles_n=cfg.triangles_n,
        encoder_depth=cfg.model.encoder_depth,
        channels_base=cfg.model.channels_base,
        channels_progression=cfg.model.channels_progression,
        hidden_dim=cfg.model.hidden_dim,
        fc_hidden_dim=cfg.model.fc_hidden_dim
    ).to(device)
    
    # Create renderer
    renderer = create_renderer(cfg.image_size, cfg.training.device)
    
    # Create loss function
    # Create loss function based on config
    if hasattr(cfg.loss, 'type') and cfg.loss.type == 'geometric_mean':
        from .loss import GeometricMeanLoss
        loss_fn = GeometricMeanLoss(
            vgg_layers=cfg.loss.vgg_layers,
            device=cfg.training.device,
            epsilon=cfg.loss.epsilon,
            min_loss_value=cfg.loss.min_loss_value,
            regularization_weight=cfg.loss.regularization_weight
        )
    else:
        # Default to CombinedLoss for backward compatibility
        loss_fn = CombinedLoss(
            alpha=cfg.loss.alpha,
            beta=cfg.loss.beta,
            gamma=cfg.loss.gamma,
            vgg_layers=cfg.loss.vgg_layers,
            device=cfg.training.device
        )
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(device=cfg.training.device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        cfg.paths.lmdb_path,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory
    )
    
    # Create optimizer
    if cfg.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            betas=tuple(cfg.optimizer.betas),
            weight_decay=cfg.optimizer.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.optimizer.type}")
    
    # Create scheduler
    total_steps = len(train_loader) * cfg.training.epochs
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = get_warmup_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Mixed precision scaler
    scaler = GradScaler('cuda', enabled=cfg.training.mixed_precision)
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Resume from checkpoint if exists
    checkpoint_path = Path(cfg.paths.checkpoint_dir) / 'latest_checkpoint.pt'
    if checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(str(checkpoint_path), model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")
        
        # Train
        train_stats = train_epoch(
            model, renderer, loss_fn, train_loader,
            optimizer, scheduler, scaler, epoch, cfg, metrics_calc
        )
        
        # Validate
        val_stats = validate(model, renderer, loss_fn, val_loader, cfg, metrics_calc)
        
        # Log epoch stats
        log_dict = {
            'epoch': epoch,
            'train/epoch_loss': train_stats['total'],
            'val/epoch_loss': val_stats['total'],
            'val/lpips': val_stats['lpips'],
            'val/ssim': val_stats['ssim'],
            'val/psnr': val_stats['psnr']
        }
        wandb.log(log_dict)
        
        print(f"Train Loss: {train_stats['total']:.4f}, "
              f"Val Loss: {val_stats['total']:.4f}, "
              f"Val LPIPS: {val_stats['lpips']:.4f}, "
              f"Val SSIM: {val_stats['ssim']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            checkpoint_path = Path(cfg.paths.checkpoint_dir) / f'model_epoch_{epoch + 1}.pt'
            save_checkpoint(model, optimizer, epoch, 
                          epoch * len(train_loader), val_stats['total'],
                          cfg, str(checkpoint_path))
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if val_stats['total'] < best_val_loss:
            best_val_loss = val_stats['total']
            best_path = Path(cfg.paths.checkpoint_dir) / 'best_model.pt'
            save_checkpoint(model, optimizer, epoch,
                          epoch * len(train_loader), val_stats['total'],
                          cfg, str(best_path))
            print(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        # Save latest checkpoint
        latest_path = Path(cfg.paths.checkpoint_dir) / 'latest_checkpoint.pt'
        save_checkpoint(model, optimizer, epoch,
                      epoch * len(train_loader), val_stats['total'],
                      cfg, str(latest_path))
        
        # Save sample images
        if (epoch + 1) % cfg.logging.sample_interval == 0:
            with torch.no_grad():
                # Get a sample batch
                sample_batch = next(iter(val_loader))
                sample_images = sample_batch['image'][:4].to(device)
                
                # Generate and render
                triangle_params = model(sample_images)
                coordinates, colors = model.get_triangle_params(triangle_params)
                rendered = renderer.render_batch(coordinates, colors)
                
                # Create comparison grid
                grid = create_comparison_grid(sample_images, rendered)
                
                # Log to W&B
                wandb.log({
                    'samples': wandb.Image(grid, caption=f"Epoch {epoch + 1}")
                })
                
                # Save locally
                sample_path = Path(cfg.paths.output_dir) / f'samples_epoch_{epoch + 1}.png'
                save_image(rendered[0], str(sample_path))
    
    print("\nTraining complete!")
    wandb.finish()