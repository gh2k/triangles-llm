import pytest
from triangulate_ai.config import (
    TriangulateConfig, load_config, validate_config, setup_paths
)
from omegaconf import OmegaConf
import tempfile
import os


class TestConfiguration:
    """Test cases for configuration system."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        cfg = OmegaConf.structured(TriangulateConfig)
        
        assert cfg.image_size == 256
        assert cfg.triangles_n == 100
        assert cfg.model.encoder_depth == 5
        assert cfg.loss.alpha == 1.0
        assert cfg.training.batch_size == 4
    
    def test_load_config_default(self):
        """Test loading configuration without file."""
        cfg = load_config()
        
        assert cfg.image_size == 256
        assert cfg.triangles_n == 100
    
    def test_load_config_with_overrides(self):
        """Test configuration with CLI overrides."""
        overrides = ['image_size=512', 'triangles_n=200', 'training.batch_size=8']
        cfg = load_config(overrides=overrides)
        
        assert cfg.image_size == 512
        assert cfg.triangles_n == 200
        assert cfg.training.batch_size == 8
    
    def test_load_config_from_file(self):
        """Test loading configuration from YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
image_size: 512
triangles_n: 150
model:
  encoder_depth: 4
loss:
  alpha: 2.0
  beta: 0.2
""")
            temp_path = f.name
        
        try:
            cfg = load_config(temp_path)
            assert cfg.image_size == 512
            assert cfg.triangles_n == 150
            assert cfg.model.encoder_depth == 4
            assert cfg.loss.alpha == 2.0
            assert cfg.loss.beta == 0.2
        finally:
            os.unlink(temp_path)
    
    def test_validate_config_valid(self):
        """Test configuration validation with valid values."""
        cfg = load_config()
        validate_config(cfg)  # Should not raise
    
    def test_validate_config_invalid_image_size(self):
        """Test configuration validation with invalid image size."""
        cfg = load_config(overrides=['image_size=0'])
        with pytest.raises(AssertionError):
            validate_config(cfg)
    
    def test_validate_config_invalid_triangles(self):
        """Test configuration validation with invalid triangle count."""
        cfg = load_config(overrides=['triangles_n=-10'])
        with pytest.raises(AssertionError):
            validate_config(cfg)
    
    def test_validate_config_invalid_loss_weights(self):
        """Test configuration validation with all zero loss weights."""
        cfg = load_config(overrides=['loss.alpha=0', 'loss.beta=0', 'loss.gamma=0'])
        with pytest.raises(AssertionError):
            validate_config(cfg)
    
    def test_setup_paths(self):
        """Test directory creation from configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = load_config(overrides=[
                f'paths.data_dir={tmpdir}/data',
                f'paths.checkpoint_dir={tmpdir}/checkpoints',
                f'paths.output_dir={tmpdir}/outputs',
                f'paths.lmdb_path={tmpdir}/data/images.lmdb'
            ])
            
            setup_paths(cfg)
            
            assert os.path.exists(f'{tmpdir}/data')
            assert os.path.exists(f'{tmpdir}/checkpoints')
            assert os.path.exists(f'{tmpdir}/outputs')