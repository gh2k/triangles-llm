import torch
import numpy as np
from typing import Tuple, Optional
import warnings

try:
    import pydiffvg
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
        self.torch_device = torch.device(device)
        
        # Set DiffVG device
        if 'cuda' in device:
            pydiffvg.set_use_gpu(True)
            if ':' in device:
                gpu_id = int(device.split(':')[1])
                torch.cuda.set_device(gpu_id)
                pydiffvg.set_device(self.torch_device)
        else:
            pydiffvg.set_use_gpu(False)
    
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
        
        # Validate and clamp coordinates to prevent NaN/inf issues
        if torch.isnan(coordinates).any():
            nan_count = torch.isnan(coordinates).sum().item()
            warnings.warn(f"NaN detected in {nan_count} coordinate values, replacing with zeros")
            coordinates = torch.where(torch.isnan(coordinates), torch.zeros_like(coordinates), coordinates)
        
        if torch.isinf(coordinates).any():
            inf_count = torch.isinf(coordinates).sum().item()
            warnings.warn(f"Inf detected in {inf_count} coordinate values, clamping to [-1, 1]")
            coordinates = torch.where(torch.isinf(coordinates), 
                                    torch.sign(coordinates).clamp(-1, 1), coordinates)
        
        coordinates = torch.clamp(coordinates, -1.0, 1.0)
        
        # Validate and clamp colors
        colors = torch.clamp(colors, 0.0, 1.0)
        if torch.isnan(colors).any() or torch.isinf(colors).any():
            warnings.warn("NaN or Inf detected in colors, replacing with defaults")
            colors = torch.where(torch.isnan(colors) | torch.isinf(colors), 
                               torch.ones_like(colors) * 0.5, colors)
        
        # Convert coordinates from [-1, 1] to [0, image_size]
        coords_scaled = (coordinates + 1.0) * 0.5 * self.image_size
        
        # Debug: check for issues in coords_scaled
        if torch.isnan(coords_scaled).any() or torch.isinf(coords_scaled).any():
            warnings.warn(f"NaN/Inf detected in coords_scaled after conversion")
        
        # Create shapes list for DiffVG
        shapes = []
        shape_groups = []
        
        valid_shape_count = 0
        for i in range(n_triangles):
            # Extract triangle vertices and ensure contiguous
            points = coords_scaled[i].reshape(3, 2).contiguous()  # (3, 2) for 3 vertices
            
            # Check for NaN/Inf in points
            if torch.isnan(points).any() or torch.isinf(points).any():
                continue
            
            # Check if triangle is valid (non-degenerate)
            p1, p2, p3 = points[0], points[1], points[2]
            area = 0.5 * torch.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - 
                                   (p3[0] - p1[0]) * (p2[1] - p1[1]))
            
            # Skip degenerate triangles
            if area < 1e-6 or torch.isnan(area) or torch.isinf(area):
                continue
            
            # Create polygon for triangle (pydiffvg expects CPU float32 tensors)
            polygon = pydiffvg.Polygon(
                points=points.cpu().float(),
                is_closed=True
            )
            shapes.append(polygon)
            
            # Create shape group with color (pydiffvg expects CPU float32 tensors)
            color_rgba = colors[i]  # (4,) tensor with RGBA
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([valid_shape_count], dtype=torch.int32),  # CPU tensor
                fill_color=color_rgba.cpu().float().contiguous()  # RGBA on CPU as float32
            )
            shape_groups.append(shape_group)
            valid_shape_count += 1
        
        # If no valid shapes, create a small default triangle to avoid empty scene
        if valid_shape_count == 0:
            warnings.warn(f"No valid triangles found out of {n_triangles}. Creating default triangle.")
            default_points = torch.tensor([[10.0, 10.0], [100.0, 10.0], [55.0, 100.0]], 
                                        dtype=torch.float32)  # CPU tensor, in pixel coordinates
            polygon = pydiffvg.Polygon(points=default_points, is_closed=True)
            shapes.append(polygon)
            
            default_color = torch.tensor([0.5, 0.5, 0.5, 0.5], 
                                       dtype=torch.float32)  # CPU tensor
            shape_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([0], dtype=torch.int32),  # CPU tensor
                fill_color=default_color
            )
            shape_groups.append(shape_group)
        
        # Serialize scene
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            self.image_size, self.image_size, shapes, shape_groups
        )
        
        # Render with white background
        # Note: background_image should be None or shape (H, W, 4)
        # We'll render with transparent background and composite white later
        render = pydiffvg.RenderFunction.apply
        rendered = render(
            self.image_size,    # width
            self.image_size,    # height
            2,                  # num_samples_x
            2,                  # num_samples_y
            0,                  # seed
            None,               # background image (None = transparent)
            *scene_args
        )
        
        # The output is (H, W, 4) with RGBA channels
        # We need to composite over white background and convert to RGB
        if rendered.shape[2] == 4:
            # Extract alpha channel
            alpha = rendered[:, :, 3:4]
            rgb = rendered[:, :, :3]
            
            # Composite over white background using alpha blending
            white_bg = torch.ones_like(rgb)
            rendered_rgb = rgb * alpha + white_bg * (1 - alpha)
            
            # Permute to (3, H, W)
            rendered = rendered_rgb.permute(2, 0, 1)
        else:
            # Already RGB, just permute
            rendered = rendered.permute(2, 0, 1)
        
        # Ensure output is in [0, 1] range
        rendered = torch.clamp(rendered, 0, 1)
        
        # Move to the correct device
        rendered = rendered.to(self.torch_device)
        
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
        self.torch_device = torch.device(device)
        warnings.warn("Using MockRenderer because DiffVG is not available. "
                     "Rendered images will be random noise.")
    
    def render_batch(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Generate random images as placeholder."""
        batch_size = coordinates.size(0)
        return torch.rand(batch_size, 3, self.image_size, self.image_size, 
                         device=self.torch_device, requires_grad=True)
    
    def render_single(self, coordinates: torch.Tensor, colors: torch.Tensor) -> torch.Tensor:
        """Generate random image as placeholder."""
        return torch.rand(3, self.image_size, self.image_size, 
                         device=self.torch_device, requires_grad=True)
    
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