# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TriangulateAI is a Python-based neural network application that converts RGB images into stylized approximations using translucent triangles. The project outputs both PNG and SVG formats using differentiable vector rendering.

## Development Commands

### Setup & Dependencies
```bash
# Install dependencies (once implemented)
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Training
```bash
# Train model with default config
triangulate_ai train --config config.yaml

# Train with custom parameters
triangulate_ai train --config config.yaml --triangles_n 200 --device cuda:0
```

### Inference
```bash
# Generate PNG and SVG from trained model
triangulate_ai infer model.pt --input img.jpg --output out.png --svg out.svg

# Inference with metrics against target
triangulate_ai infer model.pt --input img.jpg --output out.png --target original.jpg
```

### Data Preprocessing
```bash
# Convert images to LMDB format for training
triangulate_ai preprocess --input-dir images/ --output-db data.lmdb
```

### Evaluation
```bash
# Run metrics on validation set
triangulate_ai eval model.pt --dataset val/ --metrics lpips,ssim,psnr
```

### Code Quality
```bash
# Run linting (once setup.cfg/pyproject.toml configured)
flake8 triangulate_ai/
mypy triangulate_ai/

# Run tests
pytest tests/
pytest tests/test_specific.py -v  # Run specific test file
```

## Architecture Overview

The system follows an end-to-end differentiable pipeline:

1. **Image Input** → **Preprocessor** (resize, normalize, augment) → **LMDB Storage**
2. **Training Loop**: LMDB → **Generator Network** (CNN encoder → FC decoder) → **Triangle Parameters** (n×10)
3. **Rendering**: Triangle Parameters → **DiffVG Renderer** → **Rendered Image**
4. **Loss Computation**: Rendered vs Original using Perceptual (VGG19), L1, and LPIPS losses
5. **Inference**: Model → Triangle Parameters → **PNG Export** + **SVG Export**

### Key Components

- **Generator Network**: Configurable CNN encoder (default 5 blocks) with FC decoder outputting n×10 parameters per image
  - First 6 values per triangle: coordinates (x1,y1,x2,y2,x3,y3) with tanh activation [-1,1]
  - Last 4 values per triangle: RGBA color with sigmoid activation [0,1]

- **DiffVG Renderer**: Differentiable rasterization of triangles with alpha compositing
  - Triangles rendered in index order (0→n-1)
  - White background with "over" alpha blending

- **Loss Module**: Weighted combination of three losses
  - Perceptual: VGG19 features at layers conv1_2, conv2_2, conv3_4, conv4_4
  - Pixel: L1 distance
  - LPIPS: Learned perceptual metric

### Configuration System

All hyperparameters are managed through YAML config with CLI overrides:
- `config.yaml` contains all default parameters
- CLI flags override YAML values (e.g., `--triangles_n 200`)
- Key parameters: `image_size`, `triangles_n`, `encoder_depth`, loss weights (α, β, γ)

## Performance Targets

- Training: ≥25 iterations/second at 256×256, batch=4, n=100
- GPU Memory: <28GB during training, <8GB during inference
- Inference: <1 second for 512×512 image with 100 triangles
- Quality: LPIPS ≤0.18 and SSIM ≥0.72 on validation set

## Known Limitations

- **Mixed Precision Training**: DiffVG requires float32 for numerical stability. Mixed precision (float16) is disabled by default as it causes assertion errors in the backward pass.

## Dependencies

- Python ≥3.10
- PyTorch ≥2.1
- CUDA ≥12.2
- DiffVG (differentiable rendering)
- torchvision (VGG19 for perceptual loss)
- LPIPS
- OmegaConf/Hydra (configuration)
- Weights & Biases (experiment tracking)