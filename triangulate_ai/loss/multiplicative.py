import torch
import torch.nn as nn
from typing import Dict
from .perceptual import VGGPerceptualLoss, CombinedLoss
import lpips


class MultiplicativeLoss(nn.Module):
    """Experimental multiplicative loss combination with safety measures."""
    
    def __init__(self,
                 vgg_layers=None,
                 device='cuda',
                 epsilon=1e-4,
                 normalize=True,
                 use_log_space=False):
        """
        Initialize multiplicative loss.
        
        Args:
            vgg_layers: VGG layers for perceptual loss
            device: Device to run on
            epsilon: Small value to prevent zero losses
            normalize: Whether to normalize losses to [0,1] range
            use_log_space: Use log-space addition (more stable)
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.normalize = normalize
        self.use_log_space = use_log_space
        self.device = device
        
        # Initialize loss components
        self.perceptual_loss = VGGPerceptualLoss(vgg_layers, device)
        self.l1_loss = nn.L1Loss()
        self.lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
        
        # Running statistics for normalization
        self.register_buffer('perceptual_mean', torch.tensor(1.0))
        self.register_buffer('perceptual_std', torch.tensor(1.0))
        self.register_buffer('l1_mean', torch.tensor(1.0))
        self.register_buffer('l1_std', torch.tensor(1.0))
        self.register_buffer('lpips_mean', torch.tensor(1.0))
        self.register_buffer('lpips_std', torch.tensor(1.0))
        self.register_buffer('update_count', torch.tensor(0))
        
    def update_stats(self, perceptual, l1, lpips_val):
        """Update running statistics for normalization."""
        momentum = 0.99
        
        if self.update_count < 100:
            # Initial phase: use simple average
            alpha = 1.0 / (self.update_count + 1)
            self.perceptual_mean = alpha * perceptual + (1 - alpha) * self.perceptual_mean
            self.l1_mean = alpha * l1 + (1 - alpha) * self.l1_mean
            self.lpips_mean = alpha * lpips_val + (1 - alpha) * self.lpips_mean
        else:
            # After warmup: use momentum
            self.perceptual_mean = momentum * self.perceptual_mean + (1 - momentum) * perceptual
            self.l1_mean = momentum * self.l1_mean + (1 - momentum) * l1
            self.lpips_mean = momentum * self.lpips_mean + (1 - momentum) * lpips_val
        
        self.update_count += 1
    
    def forward(self, rendered, target):
        """Compute multiplicative loss with safety measures."""
        losses = {}
        
        # Compute individual losses
        perceptual = self.perceptual_loss(rendered, target)
        l1 = self.l1_loss(rendered, target)
        
        # LPIPS expects [-1, 1]
        rendered_lpips = 2 * rendered - 1
        target_lpips = 2 * target - 1
        lpips_val = self.lpips_loss(rendered_lpips, target_lpips).mean()
        
        # Store raw values
        losses['perceptual'] = perceptual
        losses['l1'] = l1
        losses['lpips'] = lpips_val
        
        # Update statistics (only in training mode)
        if self.training:
            with torch.no_grad():
                self.update_stats(perceptual.item(), l1.item(), lpips_val.item())
        
        # Normalize if requested
        if self.normalize and self.update_count > 10:
            perceptual_norm = (perceptual / (self.perceptual_mean + 1e-8)).clamp(0, 10)
            l1_norm = (l1 / (self.l1_mean + 1e-8)).clamp(0, 10)
            lpips_norm = (lpips_val / (self.lpips_mean + 1e-8)).clamp(0, 10)
        else:
            perceptual_norm = perceptual
            l1_norm = l1
            lpips_norm = lpips_val
        
        # Add epsilon to prevent zeros
        perceptual_safe = perceptual_norm + self.epsilon
        l1_safe = l1_norm + self.epsilon
        lpips_safe = lpips_norm + self.epsilon
        
        if self.use_log_space:
            # Log-space addition (mathematically equivalent to multiplication)
            # More numerically stable
            log_total = torch.log(perceptual_safe) + torch.log(l1_safe) + torch.log(lpips_safe)
            total = torch.exp(log_total)
        else:
            # Direct multiplication
            total = perceptual_safe * l1_safe * lpips_safe
        
        # Gradient clipping built into the loss
        # This helps prevent gradient explosion
        total = total.clamp(max=100.0)
        
        losses['total'] = total
        losses['perceptual_norm'] = perceptual_norm
        losses['l1_norm'] = l1_norm
        losses['lpips_norm'] = lpips_norm
        
        return losses


class HybridLoss(CombinedLoss):
    """Hybrid approach: multiplicative for some terms, additive for others."""
    
    def __init__(self, 
                 alpha=1.0,
                 beta=0.1,
                 gamma=0.05,
                 multiplicative_weight=0.1,
                 vgg_layers=None,
                 device='cuda'):
        """
        Hybrid loss that combines additive and multiplicative approaches.
        
        The total loss is:
        L_total = (1-w) * L_additive + w * L_multiplicative
        
        where w is the multiplicative_weight.
        """
        super().__init__(alpha, beta, gamma, vgg_layers, device)
        self.multiplicative_weight = multiplicative_weight
        self.multiplicative_loss = MultiplicativeLoss(vgg_layers, device, normalize=True)
    
    def forward(self, rendered, target):
        """Compute hybrid loss."""
        # Get additive losses
        additive_losses = super().forward(rendered, target)
        
        # Get multiplicative losses
        mult_losses = self.multiplicative_loss(rendered, target)
        
        # Combine
        total = ((1 - self.multiplicative_weight) * additive_losses['total'] + 
                 self.multiplicative_weight * mult_losses['total'])
        
        # Return combined results
        losses = {
            'total': total,
            'additive_total': additive_losses['total'],
            'multiplicative_total': mult_losses['total'],
            'perceptual': additive_losses['perceptual'],
            'l1': additive_losses['l1'],
            'lpips': additive_losses['lpips']
        }
        
        return losses