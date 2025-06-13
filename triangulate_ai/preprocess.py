import os
import lmdb
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm
import random
from omegaconf import DictConfig


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.tiff', '.bmp')) -> List[Path]:
    """Get all image files from a directory recursively."""
    image_files = []
    path = Path(directory)
    
    for ext in extensions:
        image_files.extend(path.rglob(f'*{ext}'))
        image_files.extend(path.rglob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def resize_with_padding(image: Image.Image, target_size: int, fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Resize image maintaining aspect ratio with padding."""
    # Calculate new dimensions
    w, h = image.size
    aspect = w / h
    
    if aspect > 1:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)
    
    # Resize image
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Create new image with padding
    padded = Image.new('RGB', (target_size, target_size), fill_color)
    
    # Calculate position to paste
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    
    padded.paste(image, (x, y))
    
    return padded


def apply_augmentations(image: Image.Image, hsv_jitter: float = 0.1, random_flip: bool = True) -> Image.Image:
    """Apply data augmentations to image."""
    # Random horizontal flip
    if random_flip and random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # HSV jitter
    if hsv_jitter > 0:
        # Convert to HSV
        hsv = image.convert('HSV')
        h, s, v = hsv.split()
        
        # Apply jitter
        h_array = np.array(h)
        s_array = np.array(s)
        v_array = np.array(v)
        
        # Hue jitter (circular)
        h_jitter = int(180 * hsv_jitter * (random.random() * 2 - 1))
        h_array = (h_array.astype(np.int16) + h_jitter) % 180
        
        # Saturation and value jitter
        s_factor = 1 + hsv_jitter * (random.random() * 2 - 1)
        v_factor = 1 + hsv_jitter * (random.random() * 2 - 1)
        
        s_array = np.clip(s_array * s_factor, 0, 255).astype(np.uint8)
        v_array = np.clip(v_array * v_factor, 0, 255).astype(np.uint8)
        
        # Reconstruct image
        hsv = Image.merge('HSV', (
            Image.fromarray(h_array.astype(np.uint8)),
            Image.fromarray(s_array),
            Image.fromarray(v_array)
        ))
        image = hsv.convert('RGB')
    
    return image


def preprocess_image(image_path: Path, target_size: int, normalize: bool = True,
                    augment: bool = False, hsv_jitter: float = 0.1, 
                    random_flip: bool = True) -> torch.Tensor:
    """Load and preprocess a single image."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize with padding
    image = resize_with_padding(image, target_size)
    
    # Apply augmentations if requested
    if augment:
        image = apply_augmentations(image, hsv_jitter, random_flip)
    
    # Convert to tensor
    tensor = transforms.ToTensor()(image)
    
    # Normalize to [0, 1] is already done by ToTensor
    
    return tensor


def preprocess_images(input_dir: str, output_db: str, cfg: DictConfig) -> None:
    """Preprocess all images in directory and save to LMDB database."""
    print(f"Preprocessing images from {input_dir} to {output_db}")
    
    # Get all image files
    image_files = get_image_files(input_dir)
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Shuffle and split into train/val
    random.shuffle(image_files)
    split_idx = int(len(image_files) * cfg.data.train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")
    
    # Create LMDB environment
    map_size = 100 * 1024 * 1024 * 1024  # 100GB max
    env = lmdb.open(output_db, map_size=map_size)
    
    # Process training images
    with env.begin(write=True) as txn:
        print("Processing training images...")
        for idx, image_path in enumerate(tqdm(train_files)):
            try:
                tensor = preprocess_image(
                    image_path,
                    cfg.image_size,
                    normalize=True,
                    augment=True,
                    hsv_jitter=cfg.data.augmentation.hsv_jitter,
                    random_flip=cfg.data.augmentation.random_flip
                )
                
                # Serialize tensor
                data = {
                    'image': tensor.numpy(),
                    'path': str(image_path),
                    'shape': tensor.shape
                }
                serialized = pickle.dumps(data)
                
                # Store in LMDB
                key = f'train_{idx:08d}'.encode()
                txn.put(key, serialized)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Process validation images (no augmentation)
        print("Processing validation images...")
        for idx, image_path in enumerate(tqdm(val_files)):
            try:
                tensor = preprocess_image(
                    image_path,
                    cfg.image_size,
                    normalize=True,
                    augment=False
                )
                
                # Serialize tensor
                data = {
                    'image': tensor.numpy(),
                    'path': str(image_path),
                    'shape': tensor.shape
                }
                serialized = pickle.dumps(data)
                
                # Store in LMDB
                key = f'val_{idx:08d}'.encode()
                txn.put(key, serialized)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
        
        # Store metadata
        metadata = {
            'train_count': len(train_files),
            'val_count': len(val_files),
            'image_size': cfg.image_size,
            'total_count': len(image_files)
        }
        txn.put(b'__metadata__', pickle.dumps(metadata))
    
    env.close()
    print(f"Preprocessing complete. Database saved to {output_db}")