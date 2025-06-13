import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from typing import Dict, List, Optional


class MetricsCalculator:
    """Calculate evaluation metrics between images."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='vgg').to(device)
        self.lpips_fn.eval()
        
        for param in self.lpips_fn.parameters():
            param.requires_grad = False
    
    def calculate_metrics(self, rendered: torch.Tensor, target: torch.Tensor,
                         metrics: List[str] = ['lpips', 'ssim', 'psnr']) -> Dict[str, float]:
        """
        Calculate metrics between rendered and target images.
        
        Args:
            rendered: Rendered images (B, 3, H, W) in range [0, 1]
            target: Target images (B, 3, H, W) in range [0, 1]
            metrics: List of metrics to calculate
            
        Returns:
            Dictionary of metric values
        """
        results = {}
        
        # Convert to numpy for some metrics
        rendered_np = rendered.detach().cpu().numpy().transpose(0, 2, 3, 1)  # (B, H, W, 3)
        target_np = target.detach().cpu().numpy().transpose(0, 2, 3, 1)
        
        batch_size = rendered.shape[0]
        
        if 'lpips' in metrics:
            # LPIPS expects [-1, 1]
            rendered_lpips = 2 * rendered - 1
            target_lpips = 2 * target - 1
            lpips_vals = []
            
            for i in range(batch_size):
                val = self.lpips_fn(rendered_lpips[i:i+1], target_lpips[i:i+1])
                lpips_vals.append(val.item())
            
            results['lpips'] = np.mean(lpips_vals)
        
        if 'ssim' in metrics:
            ssim_vals = []
            for i in range(batch_size):
                val = ssim(target_np[i], rendered_np[i], 
                          channel_axis=2, data_range=1.0)
                ssim_vals.append(val)
            results['ssim'] = np.mean(ssim_vals)
        
        if 'psnr' in metrics:
            psnr_vals = []
            for i in range(batch_size):
                val = psnr(target_np[i], rendered_np[i], data_range=1.0)
                psnr_vals.append(val)
            results['psnr'] = np.mean(psnr_vals)
        
        return results