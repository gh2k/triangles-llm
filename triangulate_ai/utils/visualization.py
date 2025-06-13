import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import io


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy image.
    
    Args:
        tensor: Image tensor (3, H, W) or (B, 3, H, W) in range [0, 1]
        
    Returns:
        Numpy array (H, W, 3) or (B, H, W, 3) in range [0, 255]
    """
    if tensor.dim() == 4:
        # Batch dimension
        images = []
        for i in range(tensor.size(0)):
            img = tensor[i].detach().cpu().numpy()
            img = np.transpose(img, (1, 2, 0))  # CHW to HWC
            img = (img * 255).astype(np.uint8)
            images.append(img)
        return np.array(images)
    else:
        # Single image
        img = tensor.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # CHW to HWC
        img = (img * 255).astype(np.uint8)
        return img


def save_image(tensor: torch.Tensor, path: str) -> None:
    """Save tensor as image file."""
    img_array = tensor_to_image(tensor)
    img = Image.fromarray(img_array)
    img.save(path)


def create_comparison_grid(original: torch.Tensor, 
                          rendered: torch.Tensor,
                          titles: Optional[List[str]] = None) -> np.ndarray:
    """
    Create a side-by-side comparison grid.
    
    Args:
        original: Original images (B, 3, H, W)
        rendered: Rendered images (B, 3, H, W)
        titles: Optional titles for images
        
    Returns:
        Grid image as numpy array
    """
    batch_size = original.size(0)
    
    if titles is None:
        titles = ['Original', 'Rendered'] * batch_size
    
    fig, axes = plt.subplots(batch_size, 2, figsize=(10, 5 * batch_size))
    
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # Original
        orig_img = tensor_to_image(original[i])
        axes[i, 0].imshow(orig_img)
        axes[i, 0].set_title(titles[i * 2])
        axes[i, 0].axis('off')
        
        # Rendered
        rend_img = tensor_to_image(rendered[i])
        axes[i, 1].imshow(rend_img)
        axes[i, 1].set_title(titles[i * 2 + 1])
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img_array = np.array(img)
    plt.close()
    
    return img_array


def visualize_triangles(coordinates: torch.Tensor, 
                       colors: torch.Tensor,
                       image_size: int,
                       background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Visualize triangles as a simple rasterization (for debugging).
    
    Args:
        coordinates: Triangle coordinates (N, 6) in range [-1, 1]
        colors: Triangle colors (N, 4) RGBA in range [0, 1]
        image_size: Output image size
        background_color: Background RGB color
        
    Returns:
        Image as numpy array (H, W, 3)
    """
    from PIL import Image, ImageDraw
    
    # Create image
    img = Image.new('RGB', (image_size, image_size), background_color)
    draw = ImageDraw.Draw(img, 'RGBA')
    
    # Convert coordinates
    coords_scaled = (coordinates + 1.0) * 0.5 * image_size
    
    # Draw triangles
    for i in range(coordinates.size(0)):
        points = coords_scaled[i].detach().cpu().numpy()
        x1, y1, x2, y2, x3, y3 = points
        
        color = colors[i].detach().cpu().numpy()
        r, g, b, a = color
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        a = int(a * 255)
        
        # Draw filled triangle
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], 
                    fill=(r, g, b, a))
    
    return np.array(img)