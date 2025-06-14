import torch
import torch.nn as nn
from typing import Dict, Optional
from .perceptual import VGGPerceptualLoss
import lpips


class GeometricMeanLoss(nn.Module):
    """
    Geometric mean loss implementation using log-sum approach.
    
    This loss combines multiple objectives using their geometric mean,
    which inherently balances all losses without manual weight tuning.
    The log-sum implementation avoids numerical issues with direct multiplication.
    
    Total loss = exp(1/n * sum(log(L_i + eps)))
    
    In practice, we optimize sum(log(L_i + eps)) which has the same minima.
    """
    
    def __init__(self,
                 vgg_layers=None,
                 device='cuda',
                 epsilon=1e-8,
                 min_loss_value=1e-4,
                 regularization_weight=0.01):
        """
        Initialize geometric mean loss.
        
        Args:
            vgg_layers: VGG layers for perceptual loss
            device: Device to run on
            epsilon: Small value to prevent log(0)
            min_loss_value: Minimum value for L1 regularization
            regularization_weight: Weight for L1 regularization term
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.min_loss_value = min_loss_value
        self.regularization_weight = regularization_weight
        self.device = device
        
        # Initialize loss components
        self.perceptual_loss = VGGPerceptualLoss(vgg_layers, device)
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
            
    def compute_regularization(self, perceptual, l1, lpips_val):
        """
        Compute L1 regularization to prevent losses from getting too close to zero.
        
        This helps avoid the vanishing gradient problem when all losses become small,
        and prevents any single loss from dominating by becoming extremely small.
        """
        # Penalize losses that are below the minimum threshold
        reg = torch.tensor(0.0, device=self.device, dtype=perceptual.dtype)
        
        if l1 < self.min_loss_value:
            reg = reg + (self.min_loss_value - l1)**2
            
        if perceptual < self.min_loss_value:
            reg = reg + (self.min_loss_value - perceptual)**2
            
        if lpips_val < self.min_loss_value:
            reg = reg + (self.min_loss_value - lpips_val)**2
            
        return reg * self.regularization_weight
    
    def forward(self, rendered, target):
        """
        Compute geometric mean loss using log-sum approach.
        
        Args:
            rendered: Rendered images (B, 3, H, W) in range [0, 1]
            target: Target images (B, 3, H, W) in range [0, 1]
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Compute individual losses
        perceptual = self.perceptual_loss(rendered, target)
        l1 = self.l1_loss(rendered, target)
        
        # LPIPS expects [-1, 1]
        rendered_lpips = 2 * rendered - 1
        target_lpips = 2 * target - 1
        lpips_val = self.lpips_loss(rendered_lpips, target_lpips).mean()
        
        # Store raw values for monitoring
        losses['perceptual'] = perceptual
        losses['l1'] = l1
        losses['lpips'] = lpips_val
        
        # Compute log of each loss (with epsilon for stability)
        log_perceptual = torch.log(perceptual + self.epsilon)
        log_l1 = torch.log(l1 + self.epsilon)
        log_lpips = torch.log(lpips_val + self.epsilon)
        
        # Sum of logs (equivalent to log of product)
        log_sum = log_perceptual + log_l1 + log_lpips
        
        # Add regularization to prevent losses from getting too small
        regularization = self.compute_regularization(
            perceptual, 
            l1, 
            lpips_val
        )
        
        # Total loss is the sum of logs plus regularization
        # Note: We don't divide by 3 because that's just a constant factor
        # that doesn't affect optimization
        total = log_sum + regularization
        
        # Store additional values for monitoring
        losses['total'] = total
        losses['log_perceptual'] = log_perceptual
        losses['log_l1'] = log_l1
        losses['log_lpips'] = log_lpips
        losses['regularization'] = regularization
        
        # Compute the actual geometric mean for monitoring
        # (not used for optimization, just for logging)
        geometric_mean = torch.exp(log_sum / 3.0)
        losses['geometric_mean'] = geometric_mean
        
        return losses


class AdaptiveGeometricMeanLoss(GeometricMeanLoss):
    """
    Extended version with adaptive regularization based on training progress.
    
    Early in training, we want to allow losses to decrease freely.
    Later, we increase regularization to prevent any loss from becoming too small.
    """
    
    def __init__(self,
                 vgg_layers=None,
                 device='cuda',
                 epsilon=1e-8,
                 min_loss_value=1e-4,
                 initial_regularization=0.001,
                 final_regularization=0.1,
                 warmup_epochs=50):
        """
        Initialize adaptive geometric mean loss.
        
        Args:
            initial_regularization: Regularization weight at start
            final_regularization: Regularization weight at end
            warmup_epochs: Number of epochs to ramp up regularization
        """
        super().__init__(
            vgg_layers=vgg_layers,
            device=device,
            epsilon=epsilon,
            min_loss_value=min_loss_value,
            regularization_weight=initial_regularization
        )
        
        self.initial_regularization = initial_regularization
        self.final_regularization = final_regularization
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
    def update_epoch(self, epoch):
        """Update regularization weight based on current epoch."""
        self.current_epoch = epoch
        
        # Linear ramp from initial to final regularization
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            self.regularization_weight = (
                self.initial_regularization + 
                (self.final_regularization - self.initial_regularization) * progress
            )
        else:
            self.regularization_weight = self.final_regularization