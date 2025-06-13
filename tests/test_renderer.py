import pytest
import torch
from triangulate_ai.renderer import MockRenderer, create_renderer


class TestRenderer:
    """Test cases for the renderer."""
    
    def test_mock_renderer_creation(self):
        """Test creating mock renderer."""
        renderer = MockRenderer(image_size=256, device='cpu')
        assert renderer.image_size == 256
    
    def test_mock_renderer_single(self):
        """Test rendering single image with mock renderer."""
        renderer = MockRenderer(image_size=128, device='cpu')
        
        # Create dummy triangle parameters
        coordinates = torch.randn(10, 6)  # 10 triangles
        colors = torch.rand(10, 4)  # RGBA colors
        
        # Render
        image = renderer.render_single(coordinates, colors)
        
        assert image.shape == (3, 128, 128)
        assert image.min() >= 0.0
        assert image.max() <= 1.0
    
    def test_mock_renderer_batch(self):
        """Test batch rendering with mock renderer."""
        renderer = MockRenderer(image_size=256, device='cpu')
        
        # Create batch of triangle parameters
        batch_size = 4
        n_triangles = 20
        coordinates = torch.randn(batch_size, n_triangles, 6)
        colors = torch.rand(batch_size, n_triangles, 4)
        
        # Render batch
        images = renderer.render_batch(coordinates, colors)
        
        assert images.shape == (batch_size, 3, 256, 256)
        assert images.min() >= 0.0
        assert images.max() <= 1.0
    
    def test_svg_export(self):
        """Test SVG export functionality."""
        renderer = MockRenderer(image_size=512, device='cpu')
        
        # Create triangle parameters
        coordinates = torch.tensor([
            [-0.5, -0.5, 0.5, -0.5, 0.0, 0.5],  # Triangle 1
            [0.0, 0.0, 0.3, 0.3, -0.3, 0.3]     # Triangle 2
        ])
        colors = torch.tensor([
            [1.0, 0.0, 0.0, 0.5],  # Red with 50% opacity
            [0.0, 0.0, 1.0, 0.8]   # Blue with 80% opacity
        ])
        
        # Generate SVG
        svg = renderer.triangles_to_svg(coordinates, colors)
        
        assert isinstance(svg, str)
        assert '<svg' in svg
        assert 'width="512"' in svg
        assert 'height="512"' in svg
        assert '<polygon' in svg
        assert 'rgba(' in svg
    
    def test_create_renderer_fallback(self):
        """Test renderer creation with fallback to mock."""
        # Should create MockRenderer since DiffVG is not installed
        renderer = create_renderer(image_size=256, device='cpu')
        assert isinstance(renderer, MockRenderer)