import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path
import torch

from triangulate_ai.preprocess import (
    get_image_files, resize_with_padding, apply_augmentations, 
    preprocess_image
)


class TestPreprocessing:
    """Test cases for image preprocessing."""
    
    def create_test_image(self, size=(100, 150), color=(255, 0, 0)):
        """Create a test image."""
        return Image.new('RGB', size, color)
    
    def test_get_image_files(self):
        """Test finding image files in directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for i in range(3):
                img = self.create_test_image()
                img.save(os.path.join(tmpdir, f'test_{i}.jpg'))
            
            # Create non-image file
            with open(os.path.join(tmpdir, 'test.txt'), 'w') as f:
                f.write('not an image')
            
            # Find images
            images = get_image_files(tmpdir)
            assert len(images) == 3
            assert all(f.suffix == '.jpg' for f in images)
    
    def test_resize_with_padding_landscape(self):
        """Test resizing landscape image with padding."""
        # Create landscape image (200x100)
        img = self.create_test_image(size=(200, 100), color=(255, 0, 0))
        
        # Resize to 256x256
        resized = resize_with_padding(img, 256)
        
        assert resized.size == (256, 256)
        
        # Check that red content is centered
        arr = np.array(resized)
        # Top and bottom should have padding (white)
        assert np.all(arr[0, :, :] == 255)  # Top row is white
        assert np.all(arr[-1, :, :] == 255)  # Bottom row is white
    
    def test_resize_with_padding_portrait(self):
        """Test resizing portrait image with padding."""
        # Create portrait image (100x200)
        img = self.create_test_image(size=(100, 200), color=(0, 255, 0))
        
        # Resize to 256x256
        resized = resize_with_padding(img, 256)
        
        assert resized.size == (256, 256)
        
        # Check that green content is centered
        arr = np.array(resized)
        # Left and right should have padding (white)
        assert np.all(arr[:, 0, :] == 255)  # Left column is white
        assert np.all(arr[:, -1, :] == 255)  # Right column is white
    
    def test_apply_augmentations_no_flip(self):
        """Test augmentations without flip."""
        # Create asymmetric image
        img = Image.new('RGB', (100, 100))
        pixels = img.load()
        # Make left half red, right half blue
        for y in range(100):
            for x in range(50):
                pixels[x, y] = (255, 0, 0)
            for x in range(50, 100):
                pixels[x, y] = (0, 0, 255)
        
        # Apply augmentations with no flip
        augmented = apply_augmentations(img, hsv_jitter=0.0, random_flip=False)
        
        # Check that image is unchanged
        arr_orig = np.array(img)
        arr_aug = np.array(augmented)
        assert np.array_equal(arr_orig, arr_aug)
    
    def test_preprocess_image(self):
        """Test full image preprocessing pipeline."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            img = self.create_test_image(size=(150, 100))
            img.save(f.name)
            temp_path = Path(f.name)
        
        try:
            # Preprocess without augmentation
            tensor = preprocess_image(temp_path, target_size=256, augment=False)
            
            assert isinstance(tensor, torch.Tensor)
            assert tensor.shape == (3, 256, 256)
            assert tensor.min() >= 0.0
            assert tensor.max() <= 1.0
            
            # Preprocess with augmentation
            tensor_aug = preprocess_image(temp_path, target_size=256, augment=True)
            assert tensor_aug.shape == (3, 256, 256)
            
        finally:
            os.unlink(temp_path)