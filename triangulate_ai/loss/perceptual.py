import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Dict
import lpips


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    
    def __init__(self, layer_names: List[str] = None, device: str = 'cuda'):
        """
        Initialize VGG perceptual loss.
        
        Args:
            layer_names: List of VGG layer names to extract features from
            device: Device to run on
        """
        super().__init__()
        
        if layer_names is None:
            layer_names = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4']
        
        self.layer_names = layer_names
        self.device = device
        
        # Load pretrained VGG19
        vgg19 = models.vgg19(pretrained=True).features.to(device).eval()
        
        # Freeze VGG parameters
        for param in vgg19.parameters():
            param.requires_grad = False
        
        # Create layer name to index mapping
        self.layer_mapping = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }
        
        # Build feature extractors
        self.feature_extractors = nn.ModuleDict()
        for layer_name in layer_names:
            if layer_name not in self.layer_mapping:
                raise ValueError(f"Unknown layer name: {layer_name}")
            
            layer_idx = self.layer_mapping[layer_name] + 1  # +1 to include the layer
            self.feature_extractors[layer_name] = vgg19[:layer_idx]
        
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if device != 'cpu':
            mean = mean.to(device)
            std = std.to(device)
            
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using ImageNet statistics."""
        return (x - self.mean) / self.std
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between rendered and target images.
        
        Args:
            rendered: Rendered images (B, 3, H, W) in range [0, 1]
            target: Target images (B, 3, H, W) in range [0, 1]
            
        Returns:
            Perceptual loss (scalar)
        """
        # Normalize inputs
        rendered_norm = self.normalize(rendered)
        target_norm = self.normalize(target)
        
        total_loss = 0.0
        
        # Extract features and compute L2 distance for each layer
        for layer_name, extractor in self.feature_extractors.items():
            rendered_features = extractor(rendered_norm)
            target_features = extractor(target_norm)
            
            # L2 loss on features
            layer_loss = nn.functional.mse_loss(rendered_features, target_features)
            total_loss = total_loss + layer_loss
        
        # Average over layers
        total_loss = total_loss / len(self.layer_names)
        
        return total_loss


class CombinedLoss(nn.Module):
    """Combined loss module with perceptual, L1, and LPIPS losses."""
    
    def __init__(self, 
                 alpha: float = 1.0,
                 beta: float = 0.1, 
                 gamma: float = 0.05,
                 vgg_layers: List[str] = None,
                 device: str = 'cuda'):
        """
        Initialize combined loss.
        
        Args:
            alpha: Weight for perceptual (VGG) loss
            beta: Weight for L1 pixel loss
            gamma: Weight for LPIPS loss
            vgg_layers: VGG layers to use for perceptual loss
            device: Device to run on
        """
        super().__init__()
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.device = device
        
        # Initialize loss components
        self.perceptual_loss = VGGPerceptualLoss(vgg_layers, device)
        self.l1_loss = nn.L1Loss()
        
        # Initialize LPIPS
        self.lpips_loss = lpips.LPIPS(net='vgg', spatial=False).to(device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False
    
    def forward(self, rendered: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss.
        
        Args:
            rendered: Rendered images (B, 3, H, W) in range [0, 1]
            target: Target images (B, 3, H, W) in range [0, 1]
            
        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}
        
        # Perceptual loss
        if self.alpha > 0:
            perceptual = self.perceptual_loss(rendered, target)
            losses['perceptual'] = perceptual
        else:
            losses['perceptual'] = torch.tensor(0.0, device=self.device)
        
        # L1 loss
        if self.beta > 0:
            l1 = self.l1_loss(rendered, target)
            losses['l1'] = l1
        else:
            losses['l1'] = torch.tensor(0.0, device=self.device)
        
        # LPIPS loss
        if self.gamma > 0:
            # LPIPS expects input in [-1, 1]
            rendered_lpips = 2 * rendered - 1
            target_lpips = 2 * target - 1
            lpips_val = self.lpips_loss(rendered_lpips, target_lpips).mean()
            losses['lpips'] = lpips_val
        else:
            losses['lpips'] = torch.tensor(0.0, device=self.device)
        
        # Combined loss
        total = (self.alpha * losses['perceptual'] + 
                self.beta * losses['l1'] + 
                self.gamma * losses['lpips'])
        
        losses['total'] = total
        
        return losses