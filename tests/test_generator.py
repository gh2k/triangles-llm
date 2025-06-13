import pytest
import torch
from triangulate_ai.models import TriangleGenerator


class TestTriangleGenerator:
    """Test cases for the TriangleGenerator model."""
    
    def test_model_creation(self):
        """Test model instantiation with default parameters."""
        model = TriangleGenerator(triangles_n=50)
        assert model.triangles_n == 50
        assert model.encoder_depth == 5
    
    def test_model_creation_custom(self):
        """Test model instantiation with custom parameters."""
        model = TriangleGenerator(
            triangles_n=100,
            encoder_depth=3,
            channels_base=32,
            channels_progression=[32, 64, 128],
            hidden_dim=256,
            fc_hidden_dim=512
        )
        assert model.triangles_n == 100
        assert model.encoder_depth == 3
        assert model.channels_progression == [32, 64, 128]
    
    def test_forward_pass(self):
        """Test forward pass with different batch sizes."""
        model = TriangleGenerator(triangles_n=10)
        model.eval()
        
        # Test different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, 3, 256, 256)
            with torch.no_grad():
                output = model(x)
            
            # Check output shape
            assert output.shape == (batch_size, 10, 10)
            
            # Check coordinate values are in [-1, 1]
            coords = output[:, :, :6]
            assert torch.all(coords >= -1.0)
            assert torch.all(coords <= 1.0)
            
            # Check color values are in [0, 1]
            colors = output[:, :, 6:]
            assert torch.all(colors >= 0.0)
            assert torch.all(colors <= 1.0)
    
    def test_get_triangle_params(self):
        """Test splitting output into coordinates and colors."""
        model = TriangleGenerator(triangles_n=20)
        
        # Create dummy output
        batch_size = 2
        output = torch.randn(batch_size, 20, 10)
        
        # Apply activations manually
        output[:, :, :6] = torch.tanh(output[:, :, :6])
        output[:, :, 6:] = torch.sigmoid(output[:, :, 6:])
        
        # Split parameters
        coords, colors = model.get_triangle_params(output)
        
        assert coords.shape == (batch_size, 20, 6)
        assert colors.shape == (batch_size, 20, 4)
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = TriangleGenerator(triangles_n=5)
        model.train()
        
        x = torch.randn(1, 3, 128, 128, requires_grad=True)
        output = model(x)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        
        # Check that model parameters have gradients
        for param in model.parameters():
            assert param.grad is not None
    
    def test_different_image_sizes(self):
        """Test model with different input image sizes."""
        model = TriangleGenerator(triangles_n=10)
        model.eval()
        
        # Test different image sizes
        for size in [128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (1, 10, 10)