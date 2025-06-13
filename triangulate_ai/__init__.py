"""
TriangulateAI - Neural network for image-to-triangle stylization.

This package provides an end-to-end differentiable pipeline for converting
RGB images into stylized approximations using translucent triangles.
"""

__version__ = "1.0.0"

from .models import TriangleGenerator
from .renderer import DiffVGRenderer, MockRenderer, create_renderer
from .loss import VGGPerceptualLoss, CombinedLoss
from .config import TriangulateConfig, load_config, validate_config

__all__ = [
    'TriangleGenerator',
    'DiffVGRenderer',
    'MockRenderer', 
    'create_renderer',
    'VGGPerceptualLoss',
    'CombinedLoss',
    'TriangulateConfig',
    'load_config',
    'validate_config'
]