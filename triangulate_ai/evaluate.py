import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from omegaconf import DictConfig
from typing import List, Dict
import pandas as pd

from .models import TriangleGenerator
from .renderer import create_renderer
from .utils import MetricsCalculator, save_image, create_comparison_grid
from .inference import load_model, process_image
from .datasets import LMDBDataset


def evaluate_lmdb(model_path: str, lmdb_path: str, split: str,
                 metrics: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Evaluate model on LMDB dataset."""
    device = torch.device(cfg.evaluation.device if hasattr(cfg.evaluation, 'device') else cfg.inference.device)
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    image_size = checkpoint.get('image_size', cfg.image_size)
    
    # Create renderer
    renderer = create_renderer(image_size, device)
    
    # Create dataset
    dataset = LMDBDataset(lmdb_path, split=split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.evaluation.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(device)
    
    # Evaluate
    all_metrics = {metric: [] for metric in metrics}
    
    print(f"Evaluating on {len(dataset)} {split} images...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch['image'].to(device)
            
            # Generate and render
            triangle_params = model(images)
            coordinates, colors = model.get_triangle_params(triangle_params)
            rendered = renderer.render_batch(coordinates, colors)
            
            # Calculate metrics
            batch_metrics = metrics_calc.calculate_metrics(rendered, images, metrics)
            
            for metric, value in batch_metrics.items():
                all_metrics[metric].append(value)
    
    # Compute averages
    avg_metrics = {metric: np.mean(values) for metric, values in all_metrics.items()}
    
    return avg_metrics


def evaluate_directory(model_path: str, dataset_dir: str, 
                      metrics: List[str], cfg: DictConfig) -> Dict[str, float]:
    """Evaluate model on directory of images."""
    from glob import glob
    
    device = torch.device(cfg.evaluation.device if hasattr(cfg.evaluation, 'device') else cfg.inference.device)
    
    # Load model
    model, checkpoint = load_model(model_path, device)
    image_size = checkpoint.get('image_size', cfg.image_size)
    
    # Create renderer
    renderer = create_renderer(image_size, device)
    
    # Get image pairs (assumes matching filenames in subdirectories)
    dataset_path = Path(dataset_dir)
    
    # Try to find paired images
    if (dataset_path / 'original').exists() and (dataset_path / 'target').exists():
        # Paired directory structure
        original_dir = dataset_path / 'original'
        target_dir = dataset_path / 'target'
        
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            for orig_file in original_dir.glob(f'*{ext}'):
                target_file = target_dir / orig_file.name
                if target_file.exists():
                    image_files.append((orig_file, target_file))
    else:
        # Single directory - compare against itself
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            for img_file in dataset_path.glob(f'*{ext}'):
                image_files.append((img_file, img_file))
    
    if not image_files:
        raise ValueError(f"No valid image files found in {dataset_dir}")
    
    print(f"Found {len(image_files)} images to evaluate")
    
    # Create metrics calculator
    metrics_calc = MetricsCalculator(device)
    
    # Evaluate
    all_metrics = {metric: [] for metric in metrics}
    per_image_results = []
    
    for orig_path, target_path in tqdm(image_files):
        # Load images
        input_tensor = process_image(str(orig_path), image_size).to(device)
        target_tensor = process_image(str(target_path), image_size).to(device)
        
        # Generate and render
        with torch.no_grad():
            triangle_params = model(input_tensor)
            coordinates, colors = model.get_triangle_params(triangle_params)
            rendered = renderer.render_batch(coordinates, colors)
        
        # Calculate metrics
        image_metrics = metrics_calc.calculate_metrics(rendered, target_tensor, metrics)
        
        # Store results
        for metric, value in image_metrics.items():
            all_metrics[metric].append(value)
        
        # Store per-image results
        result = {'image': orig_path.name}
        result.update(image_metrics)
        per_image_results.append(result)
    
    # Compute statistics
    stats = {}
    for metric, values in all_metrics.items():
        stats[f'{metric}_mean'] = np.mean(values)
        stats[f'{metric}_std'] = np.std(values)
        stats[f'{metric}_min'] = np.min(values)
        stats[f'{metric}_max'] = np.max(values)
    
    return stats, per_image_results


def evaluate(model_path: str, dataset_path: str, metrics: List[str], 
            cfg: DictConfig) -> None:
    """
    Main evaluation function.
    
    Args:
        model_path: Path to trained model checkpoint
        dataset_path: Path to dataset (LMDB or directory)
        metrics: List of metrics to compute
        cfg: Configuration object
    """
    print(f"Evaluating model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Metrics: {metrics}")
    
    # Check if dataset is LMDB or directory
    dataset_path_obj = Path(dataset_path)
    
    if dataset_path.endswith('.lmdb') or (dataset_path_obj / 'data.mdb').exists():
        # LMDB dataset
        print("Detected LMDB dataset")
        
        # Evaluate on validation split
        val_metrics = evaluate_lmdb(model_path, dataset_path, 'val', metrics, cfg)
        
        print("\nValidation Results:")
        for metric, value in val_metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        # Save results
        output_file = Path(cfg.paths.output_dir) / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump({'validation': val_metrics}, f, indent=2)
        
    else:
        # Directory of images
        print("Detected directory dataset")
        
        stats, per_image_results = evaluate_directory(
            model_path, dataset_path, metrics, cfg
        )
        
        print("\nEvaluation Results:")
        for metric in metrics:
            print(f"\n{metric.upper()}:")
            print(f"  Mean: {stats[f'{metric}_mean']:.4f}")
            print(f"  Std:  {stats[f'{metric}_std']:.4f}")
            print(f"  Min:  {stats[f'{metric}_min']:.4f}")
            print(f"  Max:  {stats[f'{metric}_max']:.4f}")
        
        # Save aggregate results
        output_file = Path(cfg.paths.output_dir) / 'evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save per-image results
        df = pd.DataFrame(per_image_results)
        csv_file = Path(cfg.paths.output_dir) / 'evaluation_per_image.csv'
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"  - {output_file}")
        print(f"  - {csv_file}")
    
    # Generate sample outputs
    print("\nGenerating sample outputs...")
    
    # Load model for sample generation
    device = torch.device(cfg.evaluation.device if hasattr(cfg.evaluation, 'device') else cfg.inference.device)
    model, checkpoint = load_model(model_path, device)
    image_size = checkpoint.get('image_size', cfg.image_size)
    renderer = create_renderer(image_size, device)
    
    # Get a few sample images
    if dataset_path.endswith('.lmdb') or (dataset_path_obj / 'data.mdb').exists():
        dataset = LMDBDataset(dataset_path, split='val')
        sample_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4]
        samples = [dataset[i]['image'].unsqueeze(0).to(device) for i in sample_indices[:4]]
    else:
        # From directory
        image_files = list(dataset_path_obj.glob('*.png')) + list(dataset_path_obj.glob('*.jpg'))
        samples = []
        for i in range(min(4, len(image_files))):
            img_tensor = process_image(str(image_files[i]), image_size).to(device)
            samples.append(img_tensor)
    
    # Generate and save samples
    sample_dir = Path(cfg.paths.output_dir) / 'evaluation_samples'
    sample_dir.mkdir(exist_ok=True)
    
    for i, sample in enumerate(samples):
        with torch.no_grad():
            triangle_params = model(sample)
            coordinates, colors = model.get_triangle_params(triangle_params)
            rendered = renderer.render_batch(coordinates, colors)
        
        # Save PNG
        png_path = sample_dir / f'sample_{i}_rendered.png'
        save_image(rendered[0], str(png_path))
        
        # Save SVG
        svg_path = sample_dir / f'sample_{i}_rendered.svg'
        svg_content = renderer.triangles_to_svg(coordinates[0], colors[0])
        with open(svg_path, 'w') as f:
            f.write(svg_content)
        
        # Save original
        orig_path = sample_dir / f'sample_{i}_original.png'
        save_image(sample[0], str(orig_path))
    
    print(f"Sample outputs saved to {sample_dir}")
    print("\nEvaluation complete!")