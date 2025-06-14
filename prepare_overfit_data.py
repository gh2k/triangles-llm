#!/usr/bin/env python3
"""Prepare a proper overfitting dataset by duplicating a single image multiple times."""

import os
import shutil
from pathlib import Path
import argparse


def prepare_overfit_dataset(source_image: str, output_dir: str, num_copies: int = 10):
    """
    Create a dataset with multiple copies of the same image.
    
    Args:
        source_image: Path to the source image
        output_dir: Directory to save copies
        num_copies: Number of copies to create (should be > batch_size)
    """
    source_path = Path(source_image)
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_image}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get file extension
    ext = source_path.suffix
    
    print(f"Creating {num_copies} copies of {source_path.name}")
    
    # Create copies
    for i in range(num_copies):
        dest_path = output_path / f"image_{i:04d}{ext}"
        shutil.copy2(source_path, dest_path)
        print(f"Created: {dest_path}")
    
    print(f"\nDataset created in: {output_path}")
    print(f"Total images: {num_copies}")
    print("\nNow run:")
    print(f"triangulate_ai preprocess --input-dir {output_dir} --output-db data/overfit.lmdb")
    print("triangulate_ai train --config config_overfit.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare overfitting dataset")
    parser.add_argument("source_image", help="Path to source image")
    parser.add_argument("--output-dir", default="./overfit_data", 
                       help="Output directory (default: ./overfit_data)")
    parser.add_argument("--num-copies", type=int, default=10,
                       help="Number of copies to create (default: 10)")
    
    args = parser.parse_args()
    
    prepare_overfit_dataset(args.source_image, args.output_dir, args.num_copies)