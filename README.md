# TriangulateAI

A Python-based neural network application that converts RGB images into stylized approximations using translucent triangles. The project outputs both PNG and SVG formats using differentiable vector rendering.

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install DiffVG from source:
```bash
git clone https://github.com/BachiLi/diffvg.git
cd diffvg && python setup.py install
```

3. Install TriangulateAI in development mode:
```bash
pip install -e .
```

## Quick Start

### Training
```bash
triangulate_ai train --config config.yaml
```

### Inference
```bash
triangulate_ai infer model.pt --input img.jpg --output out.png --svg out.svg
```

### Data Preprocessing
```bash
triangulate_ai preprocess --input-dir images/ --output-db data.lmdb
```

### Evaluation
```bash
triangulate_ai eval model.pt --dataset val/ --metrics lpips,ssim,psnr
```

## Testing

### Running Tests
```bash
# Run all tests
pytest

# Run with verbose output
pytest tests/

# Run specific test file
pytest tests/test_generator.py -v

# Run excluding slow tests
pytest -m "not slow"

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run with coverage report
pytest --cov=triangulate_ai
```

### Code Quality
```bash
# Run linting
flake8 triangulate_ai/

# Run type checking
mypy triangulate_ai/

# Format code
black triangulate_ai/

# Sort imports
isort triangulate_ai/
```

## License

MIT License