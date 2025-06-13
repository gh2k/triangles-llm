import torch
from PIL import Image
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import time
from typing import Optional

from .models import TriangleGenerator
from .renderer import create_renderer
from .utils import save_image, MetricsCalculator
from .preprocess import resize_with_padding


def load_model(checkpoint_path: str, device: str = 'cuda') -> tuple:
    """
    Load model from checkpoint.
    
    Returns:
        (model, checkpoint_dict)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    if 'config' in checkpoint:
        model_cfg = checkpoint['config']['model']
        triangles_n = checkpoint['config']['triangles_n']
    else:
        # Fallback to defaults if old checkpoint format
        model_cfg = {
            'encoder_depth': 5,
            'channels_base': 64,
            'channels_progression': [64, 128, 256, 512, 512],
            'hidden_dim': 512,
            'fc_hidden_dim': 1024
        }
        triangles_n = checkpoint.get('triangles_n', 100)
    
    # Create model
    model = TriangleGenerator(
        triangles_n=triangles_n,
        encoder_depth=model_cfg.get('encoder_depth', 5),
        channels_base=model_cfg.get('channels_base', 64),
        channels_progression=model_cfg.get('channels_progression'),
        hidden_dim=model_cfg.get('hidden_dim', 512),
        fc_hidden_dim=model_cfg.get('fc_hidden_dim', 1024)
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint


def process_image(image_path: str, target_size: int) -> torch.Tensor:
    """Load and preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize with padding
    image = resize_with_padding(image, target_size)
    
    # Convert to tensor
    image_array = np.array(image).astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
    
    # Add batch dimension
    return image_tensor.unsqueeze(0)


def infer(model_path: str, input_path: str, output_path: str, 
          svg_path: Optional[str], target_path: Optional[str], 
          cfg: DictConfig) -> None:
    """
    Run inference on a single image.
    
    Args:
        model_path: Path to trained model checkpoint
        input_path: Path to input image
        output_path: Path to save output PNG
        svg_path: Optional path to save SVG
        target_path: Optional path to target image for metrics
        cfg: Configuration object
    """
    print(f"Loading model from {model_path}")
    
    # Set device
    device = torch.device(cfg.inference.device)
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    
    # Get image size from checkpoint or config
    if 'image_size' in checkpoint:
        image_size = checkpoint['image_size']
    else:
        image_size = cfg.image_size
    
    # Override triangles_n if specified in config
    if hasattr(cfg, 'triangles_n') and cfg.triangles_n != model.triangles_n:
        print(f"Warning: Model was trained with {model.triangles_n} triangles, "
              f"but config specifies {cfg.triangles_n}. Using model's value.")
    
    print(f"Model loaded: {model.triangles_n} triangles, {image_size}x{image_size} images")
    
    # Create renderer
    renderer = create_renderer(image_size, device)
    
    # Load and preprocess input image
    print(f"Processing input image: {input_path}")
    input_tensor = process_image(input_path, image_size).to(device)
    
    # Run inference
    print("Generating triangles...")
    start_time = time.time()
    
    with torch.no_grad():
        # Generate triangle parameters
        triangle_params = model(input_tensor)
        
        # Split into coordinates and colors
        coordinates, colors = model.get_triangle_params(triangle_params)
        
        # Render image
        rendered = renderer.render_batch(coordinates, colors)
    
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.3f} seconds")
    
    # Save PNG output
    print(f"Saving PNG to {output_path}")
    save_image(rendered[0], output_path)
    
    # Save SVG if requested
    if svg_path:
        print(f"Saving SVG to {svg_path}")
        svg_content = renderer.triangles_to_svg(coordinates[0], colors[0])
        with open(svg_path, 'w') as f:
            f.write(svg_content)
    
    # Calculate metrics if target provided
    if target_path:
        print(f"Calculating metrics against {target_path}")
        
        # Load target image
        target_tensor = process_image(target_path, image_size).to(device)
        
        # Calculate metrics
        metrics_calc = MetricsCalculator(device)
        metrics = metrics_calc.calculate_metrics(rendered, target_tensor)
        
        print("\nMetrics:")
        print(f"LPIPS: {metrics['lpips']:.3f}")
        print(f"SSIM: {metrics['ssim']:.3f}")
        print(f"PSNR: {metrics['psnr']:.2f}")
    
    print("\nInference complete!")


def batch_infer(model_path: str, input_dir: str, output_dir: str,
                cfg: DictConfig, save_svg: bool = False) -> None:
    """
    Run inference on a directory of images.
    
    Args:
        model_path: Path to trained model checkpoint
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        cfg: Configuration object
        save_svg: Whether to save SVG files
    """
    from glob import glob
    import os
    
    # Get all image files
    input_path = Path(input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f'*{ext}'))
        image_files.extend(input_path.glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model once
    device = torch.device(cfg.inference.device)
    model, checkpoint = load_model(model_path, device)
    image_size = checkpoint.get('image_size', cfg.image_size)
    renderer = create_renderer(image_size, device)
    
    print(f"Model loaded: {model.triangles_n} triangles, {image_size}x{image_size} images")
    
    # Process each image
    total_time = 0
    
    for img_path in image_files:
        print(f"\nProcessing {img_path.name}...")
        
        # Prepare output paths
        base_name = img_path.stem
        png_path = output_path / f"{base_name}_triangles.png"
        svg_path = output_path / f"{base_name}_triangles.svg" if save_svg else None
        
        # Load and preprocess
        input_tensor = process_image(str(img_path), image_size).to(device)
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            triangle_params = model(input_tensor)
            coordinates, colors = model.get_triangle_params(triangle_params)
            rendered = renderer.render_batch(coordinates, colors)
        
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # Save outputs
        save_image(rendered[0], str(png_path))
        
        if svg_path:
            svg_content = renderer.triangles_to_svg(coordinates[0], colors[0])
            with open(svg_path, 'w') as f:
                f.write(svg_content)
        
        print(f"Saved: {png_path.name} ({inference_time:.3f}s)")
    
    avg_time = total_time / len(image_files)
    print(f"\nBatch processing complete!")
    print(f"Average inference time: {avg_time:.3f} seconds per image")
    print(f"Total time: {total_time:.2f} seconds")