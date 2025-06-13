import torch
import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import diffvg
    DIFFVG_AVAILABLE = True
except ImportError:
    DIFFVG_AVAILABLE = False
    warnings.warn("DiffVG not found. Please install it from https://github.com/BachiLi/diffvg")


class DiffVGRenderer:
    """Differentiable renderer for triangles using DiffVG."""
    
    def __init__(self, image_size: int, device: str = 'cuda'):
        """
        Initialize the DiffVG renderer.
        
        Args:
            image_size: Size of the square output image
            device: Device to render on
        """
        if not DIFFVG_AVAILABLE:
            raise ImportError("DiffVG is required but not installed. "
                            "Please install it from https://github.com/BachiLi/diffvg")
        
        self.image_size = image_size
        self.device = device
        
        # Set DiffVG device
        if 'cuda' in device:
            diffvg.set_use_gpu(True)
            if ':' in device:
                gpu_id = int(device.split(':')[1])
                diffvg.set_device(gpu_id)
        else:
            diffvg.set_use_gpu(False)
    
    def render_batch(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        Render a batch of triangle sets.
        
        Args:
            coordinates: Triangle coordinates (B, N, 6) in range [-1, 1]
            colors: Triangle colors (B, N, 4) RGBA in range [0, 1]
            
        Returns:
            Rendered images (B, 3, H, W) in range [0, 1]
        """
        batch_size = coordinates.size(0)
        n_triangles = coordinates.size(1)
        
        rendered_images = []
        
        for b in range(batch_size):
            # Render single image
            image = self.render_single(coordinates[b], colors[b])
            rendered_images.append(image)
        
        # Stack into batch
        return torch.stack(rendered_images, dim=0)
    
    def render_single(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """
        Render a single set of triangles.
        
        Args:
            coordinates: Triangle coordinates (N, 6) in range [-1, 1]
            colors: Triangle colors (N, 4) RGBA in range [0, 1]
            
        Returns:
            Rendered image (3, H, W) in range [0, 1]
        """
        n_triangles = coordinates.size(0)
        
        # Convert coordinates from [-1, 1] to [0, image_size]
        coords_scaled = (coordinates + 1.0) * 0.5 * self.image_size
        
        # Create shapes list for DiffVG
        shapes = []
        shape_groups = []
        
        for i in range(n_triangles):
            # Extract triangle vertices
            points = coords_scaled[i].reshape(3, 2)  # (3, 2) for 3 vertices
            
            # Create path for triangle
            path = diffvg.Path(
                num_control_points=torch.zeros(3, dtype=torch.int32),  # No control points for straight lines
                points=points,
                is_closed=True,
                stroke_width=torch.tensor(0.0)
            )
            shapes.append(path)
            
            # Create shape group with color
            color_rgba = colors[i]  # (4,) tensor with RGBA
            shape_group = diffvg.ShapeGroup(
                shape_ids=torch.tensor([i], dtype=torch.int32),
                fill_color=color_rgba[:3],  # RGB only
                fill_opacity=color_rgba[3:4],  # Alpha as opacity
                use_even_odd_rule=False
            )
            shape_groups.append(shape_group)
        
        # Create scene
        scene = diffvg.Scene(
            canvas_width=self.image_size,
            canvas_height=self.image_size,
            shapes=shapes,
            shape_groups=shape_groups
        )
        
        # Render with white background
        background = torch.ones(3, device=self.device)
        
        # Render args
        render_args = diffvg.RenderFunction.serialize_scene(
            self.image_size, self.image_size, shapes, shape_groups
        )
        
        # Render
        rendered = diffvg.RenderFunction.apply(
            self.image_size,    # width
            self.image_size,    # height
            2,                  # num_samples_x
            2,                  # num_samples_y
            0,                  # seed
            background,         # background color
            *render_args
        )
        
        # Reshape to (H, W, 3) and permute to (3, H, W)
        rendered = rendered.permute(2, 0, 1)
        
        # Ensure output is in [0, 1] range
        rendered = torch.clamp(rendered, 0, 1)
        
        return rendered
    
    def triangles_to_svg(self, coordinates: torch.Tensor, colors: torch.Tensor) -> str:
        """
        Convert triangles to SVG string.
        
        Args:
            coordinates: Triangle coordinates (N, 6) in range [-1, 1]
            colors: Triangle colors (N, 4) RGBA in range [0, 1]
            
        Returns:
            SVG string
        """
        n_triangles = coordinates.size(0)
        
        # Convert coordinates from [-1, 1] to [0, image_size]
        coords_scaled = (coordinates + 1.0) * 0.5 * self.image_size
        
        # Start SVG
        svg_lines = [
            f'<svg width="{self.image_size}" height="{self.image_size}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="white"/>'
        ]
        
        # Add triangles
        for i in range(n_triangles):
            # Extract vertices
            points = coords_scaled[i].detach().cpu().numpy()
            x1, y1, x2, y2, x3, y3 = points
            
            # Extract color
            color = colors[i].detach().cpu().numpy()
            r, g, b, a = color
            
            # Convert to 0-255 range
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            
            # Create polygon
            svg_lines.append(
                f'<polygon points="{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}" '
                f'fill="rgba({r},{g},{b},{a:.3f})" />'
            )
        
        svg_lines.append('</svg>')
        
        return '\n'.join(svg_lines)


class MockRenderer:
    """Mock renderer for testing without DiffVG."""
    
    def __init__(self, image_size: int, device: str = 'cuda'):
        self.image_size = image_size
        self.device = device
        warnings.warn("Using MockRenderer because DiffVG is not available. "
                     "Rendered images will be random noise.")
    
    def render_batch(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Generate random images as placeholder."""
        batch_size = coordinates.size(0)
        return torch.rand(batch_size, 3, self.image_size, self.image_size, 
                         device=self.device, requires_grad=True)
    
    def render_single(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Generate random image as placeholder."""
        return torch.rand(3, self.image_size, self.image_size, 
                         device=self.device, requires_grad=True)
    
    def triangles_to_svg(self, coordinates: torch.Tensor, colors: torch.Tensor) -> str:
        """Convert triangles to SVG string."""
        n_triangles = coordinates.size(0)
        coords_scaled = (coordinates + 1.0) * 0.5 * self.image_size
        
        svg_lines = [
            f'<svg width="{self.image_size}" height="{self.image_size}" '
            f'xmlns="http://www.w3.org/2000/svg">',
            f'<rect width="100%" height="100%" fill="white"/>'
        ]
        
        for i in range(n_triangles):
            points = coords_scaled[i].detach().cpu().numpy()
            x1, y1, x2, y2, x3, y3 = points
            color = colors[i].detach().cpu().numpy()
            r, g, b, a = color
            r, g, b = int(r * 255), int(g * 255), int(b * 255)
            svg_lines.append(
                f'<polygon points="{x1:.2f},{y1:.2f} {x2:.2f},{y2:.2f} {x3:.2f},{y3:.2f}" '
                f'fill="rgba({r},{g},{b},{a:.3f})" />'
            )
        
        svg_lines.append('</svg>')
        return '\n'.join(svg_lines)


def create_renderer(image_size: int, device: str = 'cuda') -> 'DiffVGRenderer':
    """Create a renderer, falling back to mock if DiffVG is not available."""
    if DIFFVG_AVAILABLE:
        return DiffVGRenderer(image_size, device)
    else:
        return MockRenderer(image_size, device)