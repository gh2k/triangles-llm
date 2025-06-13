from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.config_store import ConfigStore
import os


@dataclass
class ModelConfig:
    encoder_depth: int = 5
    channels_base: int = 64
    channels_progression: List[int] = field(default_factory=lambda: [64, 128, 256, 512, 512])
    hidden_dim: int = 512
    fc_hidden_dim: int = 1024


@dataclass
class LossConfig:
    alpha: float = 1.0  # Perceptual loss weight
    beta: float = 0.1   # L1 loss weight
    gamma: float = 0.05 # LPIPS loss weight
    vgg_layers: List[str] = field(default_factory=lambda: ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4'])


@dataclass
class OptimizerConfig:
    type: str = 'AdamW'
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 50
    save_interval: int = 5
    warmup_ratio: float = 0.01
    mixed_precision: bool = True
    gradient_clip: float = 1.0
    device: str = 'cuda:0'


@dataclass
class DataConfig:
    train_split: float = 0.9
    val_split: float = 0.1
    augmentation: dict = field(default_factory=lambda: {
        'random_flip': True,
        'hsv_jitter': 0.1
    })
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class LoggingConfig:
    wandb_project: str = 'triangulate_ai'
    wandb_entity: Optional[str] = None
    log_interval: int = 50
    sample_interval: int = 1


@dataclass
class PathsConfig:
    data_dir: str = 'data'
    checkpoint_dir: str = 'checkpoints'
    output_dir: str = 'outputs'
    lmdb_path: str = 'data/images.lmdb'


@dataclass
class InferenceConfig:
    batch_size: int = 1
    device: str = 'cuda:0'


@dataclass
class EvaluationConfig:
    metrics: List[str] = field(default_factory=lambda: ['lpips', 'ssim', 'psnr'])
    batch_size: int = 8


@dataclass
class TriangulateConfig:
    image_size: int = 256
    triangles_n: int = 100
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def register_configs():
    """Register configuration schemas with Hydra."""
    cs = ConfigStore.instance()
    cs.store(name="config", node=TriangulateConfig)


def load_config(config_path: Optional[str] = None, overrides: Optional[List[str]] = None) -> DictConfig:
    """Load configuration from file with optional overrides."""
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))
        return cfg
    else:
        # Create default config
        cfg = OmegaConf.structured(TriangulateConfig)
        if overrides:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(overrides))
        return cfg


def validate_config(cfg: DictConfig) -> None:
    """Validate configuration values."""
    assert cfg.image_size > 0, "image_size must be positive"
    assert cfg.triangles_n > 0, "triangles_n must be positive"
    assert cfg.model.encoder_depth > 0, "encoder_depth must be positive"
    assert 0 <= cfg.data.train_split <= 1, "train_split must be between 0 and 1"
    assert cfg.training.batch_size > 0, "batch_size must be positive"
    assert cfg.training.epochs > 0, "epochs must be positive"
    assert cfg.loss.alpha >= 0, "loss.alpha must be non-negative"
    assert cfg.loss.beta >= 0, "loss.beta must be non-negative"
    assert cfg.loss.gamma >= 0, "loss.gamma must be non-negative"
    
    # Ensure at least one loss weight is positive
    assert cfg.loss.alpha + cfg.loss.beta + cfg.loss.gamma > 0, "At least one loss weight must be positive"


def setup_paths(cfg: DictConfig) -> None:
    """Create necessary directories based on configuration."""
    os.makedirs(cfg.paths.data_dir, exist_ok=True)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cfg.paths.lmdb_path), exist_ok=True)