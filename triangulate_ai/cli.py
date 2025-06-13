import argparse
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
import os
from pathlib import Path

from .config import load_config, validate_config, setup_paths


def main():
    """Main CLI entry point for TriangulateAI."""
    parser = argparse.ArgumentParser(
        description="TriangulateAI - Convert images to triangle approximations",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--config', type=str, default='config.yaml', 
                            help='Path to configuration file')
    train_parser.add_argument('--device', type=str, help='Override device (e.g., cuda:0)')
    train_parser.add_argument('--triangles_n', type=int, help='Override number of triangles')
    train_parser.add_argument('--image_size', type=int, help='Override image size')
    train_parser.add_argument('--batch_size', type=int, help='Override batch size')
    train_parser.add_argument('--epochs', type=int, help='Override number of epochs')
    train_parser.add_argument('overrides', nargs='*', help='Additional config overrides')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess images to LMDB')
    preprocess_parser.add_argument('--input-dir', type=str, required=True, 
                                 help='Input directory containing images')
    preprocess_parser.add_argument('--output-db', type=str, required=True,
                                 help='Output LMDB database path')
    preprocess_parser.add_argument('--config', type=str, default='config.yaml',
                                 help='Path to configuration file')
    preprocess_parser.add_argument('--image_size', type=int, help='Override image size')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference on images')
    infer_parser.add_argument('model', type=str, help='Path to trained model checkpoint')
    infer_parser.add_argument('--input', type=str, required=True, help='Input image path')
    infer_parser.add_argument('--output', type=str, required=True, help='Output PNG path')
    infer_parser.add_argument('--svg', type=str, help='Output SVG path')
    infer_parser.add_argument('--target', type=str, help='Target image for metrics')
    infer_parser.add_argument('--config', type=str, default='config.yaml',
                            help='Path to configuration file')
    infer_parser.add_argument('--triangles_n', type=int, help='Override number of triangles')
    infer_parser.add_argument('--image_size', type=int, help='Override image size')
    infer_parser.add_argument('--device', type=str, help='Override device')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', help='Evaluate model on dataset')
    eval_parser.add_argument('model', type=str, help='Path to trained model checkpoint')
    eval_parser.add_argument('--dataset', type=str, required=True, 
                           help='Path to validation dataset')
    eval_parser.add_argument('--metrics', type=str, default='lpips,ssim,psnr',
                           help='Comma-separated list of metrics')
    eval_parser.add_argument('--config', type=str, default='config.yaml',
                           help='Path to configuration file')
    eval_parser.add_argument('--batch_size', type=int, help='Override batch size')
    eval_parser.add_argument('--device', type=str, help='Override device')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Load configuration
    if hasattr(args, 'config'):
        config_path = args.config
    else:
        config_path = 'config.yaml'
    
    # Build overrides list
    overrides = []
    if hasattr(args, 'device') and args.device:
        overrides.append(f'training.device={args.device}')
        overrides.append(f'inference.device={args.device}')
    if hasattr(args, 'triangles_n') and args.triangles_n:
        overrides.append(f'triangles_n={args.triangles_n}')
    if hasattr(args, 'image_size') and args.image_size:
        overrides.append(f'image_size={args.image_size}')
    if hasattr(args, 'batch_size') and args.batch_size:
        overrides.append(f'training.batch_size={args.batch_size}')
        overrides.append(f'evaluation.batch_size={args.batch_size}')
    if hasattr(args, 'epochs') and args.epochs:
        overrides.append(f'training.epochs={args.epochs}')
    if hasattr(args, 'overrides'):
        overrides.extend(args.overrides)
    
    # Load config with overrides
    cfg = load_config(config_path, overrides)
    
    # Validate configuration
    validate_config(cfg)
    
    # Setup paths
    setup_paths(cfg)
    
    # Execute command
    if args.command == 'train':
        from .train import train
        train(cfg)
    elif args.command == 'preprocess':
        from .preprocess import preprocess_images
        preprocess_images(args.input_dir, args.output_db, cfg)
    elif args.command == 'infer':
        from .inference import infer
        infer(args.model, args.input, args.output, args.svg, args.target, cfg)
    elif args.command == 'eval':
        from .evaluate import evaluate
        metrics = args.metrics.split(',')
        evaluate(args.model, args.dataset, metrics, cfg)


if __name__ == '__main__':
    main()