import pytest
import torch
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf

from triangulate_ai.models import TriangleGenerator
from triangulate_ai.renderer import create_renderer
from triangulate_ai.loss import CombinedLoss
from triangulate_ai.config import load_config, validate_config


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    def test_full_forward_pass(self):
        """Test complete forward pass through model and renderer."""
        # Load config
        cfg = load_config(overrides=['triangles_n=10', 'image_size=128'])
        validate_config(cfg)
        
        # Create model
        model = TriangleGenerator(
            triangles_n=cfg.triangles_n,
            encoder_depth=3,  # Smaller for testing
            channels_progression=[32, 64, 128]
        )
        model.eval()
        
        # Create renderer
        renderer = create_renderer(cfg.image_size, 'cpu')
        
        # Create dummy input
        batch_size = 2
        input_images = torch.rand(batch_size, 3, cfg.image_size, cfg.image_size)
        
        # Forward pass
        with torch.no_grad():
            # Generate triangles
            triangle_params = model(input_images)
            assert triangle_params.shape == (batch_size, 10, 10)
            
            # Split parameters
            coordinates, colors = model.get_triangle_params(triangle_params)
            assert coordinates.shape == (batch_size, 10, 6)
            assert colors.shape == (batch_size, 10, 4)
            
            # Render
            rendered = renderer.render_batch(coordinates, colors)
            assert rendered.shape == (batch_size, 3, cfg.image_size, cfg.image_size)
    
    def test_loss_computation(self):
        """Test loss computation."""
        cfg = load_config()
        
        # Create loss module
        loss_fn = CombinedLoss(
            alpha=cfg.loss.alpha,
            beta=cfg.loss.beta,
            gamma=cfg.loss.gamma,
            vgg_layers=['conv1_2'],  # Just one layer for speed
            device='cpu'
        )
        
        # Create dummy images
        rendered = torch.rand(2, 3, 256, 256)
        target = torch.rand(2, 3, 256, 256)
        
        # Compute losses
        losses = loss_fn(rendered, target)
        
        assert 'total' in losses
        assert 'perceptual' in losses
        assert 'l1' in losses
        assert 'lpips' in losses
        
        assert losses['total'] > 0
        assert losses['total'].requires_grad
    
    def test_gradient_flow_integration(self):
        """Test that gradients flow through the entire pipeline."""
        cfg = load_config(overrides=['triangles_n=5', 'image_size=64'])
        
        # Create components
        model = TriangleGenerator(
            triangles_n=cfg.triangles_n,
            encoder_depth=2,
            channels_progression=[16, 32]
        )
        renderer = create_renderer(cfg.image_size, 'cpu')
        loss_fn = CombinedLoss(
            alpha=1.0, beta=0.0, gamma=0.0,  # Only perceptual loss
            vgg_layers=['conv1_2'],
            device='cpu'
        )
        
        # Create input
        input_images = torch.rand(1, 3, cfg.image_size, cfg.image_size, requires_grad=True)
        target_images = torch.rand(1, 3, cfg.image_size, cfg.image_size)
        
        # Forward pass
        triangle_params = model(input_images)
        coordinates, colors = model.get_triangle_params(triangle_params)
        rendered = renderer.render_batch(coordinates, colors)
        losses = loss_fn(rendered, target_images)
        
        # Backward pass
        losses['total'].backward()
        
        # Check gradients exist
        for param in model.parameters():
            assert param.grad is not None
            assert not torch.all(param.grad == 0)
    
    def test_checkpoint_save_load(self):
        """Test saving and loading model checkpoints."""
        from triangulate_ai.train import save_checkpoint, load_checkpoint
        
        cfg = load_config(overrides=['triangles_n=15'])
        
        # Create model and optimizer
        model = TriangleGenerator(triangles_n=cfg.triangles_n)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            temp_path = f.name
        
        try:
            save_checkpoint(
                model, optimizer, epoch=5, step=100, 
                loss=0.5, cfg=cfg, checkpoint_path=temp_path
            )
            
            # Create new model and load checkpoint
            new_model = TriangleGenerator(triangles_n=cfg.triangles_n)
            new_optimizer = torch.optim.Adam(new_model.parameters())
            
            checkpoint = load_checkpoint(temp_path, new_model, new_optimizer)
            
            assert checkpoint['epoch'] == 5
            assert checkpoint['step'] == 100
            assert checkpoint['loss'] == 0.5
            assert checkpoint['triangles_n'] == 15
            
            # Check that weights are loaded correctly
            for p1, p2 in zip(model.parameters(), new_model.parameters()):
                assert torch.allclose(p1, p2)
        
        finally:
            os.unlink(temp_path)