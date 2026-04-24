"""
train.py
========
Main Training Script — Self-Pruning Neural Network on CIFAR-10

This script executes the complete training pipeline:
    1. Loads CIFAR-10 dataset with augmentation
    2. Trains self-pruning networks with multiple λ values
    3. Evaluates sparsity-accuracy trade-offs
    4. Generates comprehensive visualizations
    5. Outputs a results summary

The script is designed to be self-contained and reproducible. All results
are saved to the `results/` directory.

Usage:
    python train.py

    Environment variables:
        EPOCHS: Number of training epochs (default: 50)
        BATCH_SIZE: Batch size (default: 128)
        SEED: Random seed for reproducibility (default: 42)

Author: Likthansh Anisetti
"""

import os
import sys
import json
import time
import random
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import load_cifar10
from src.network import SelfPruningNetwork
from src.trainer import Trainer

# Visualization imports (optional — may fail on systems with DLL restrictions)
try:
    from src.visualization import (
        plot_gate_distribution,
        plot_training_curves,
        plot_lambda_comparison,
        plot_per_layer_sparsity,
    )
    HAS_PLOTTING = True
except (ImportError, OSError):
    HAS_PLOTTING = False


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic operations (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a Self-Pruning Neural Network on CIFAR-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python train.py
    python train.py --epochs 100 --batch-size 256
    python train.py --lambdas 1e-5 1e-4 1e-3 5e-3
        """
    )
    parser.add_argument(
        '--epochs', type=int, default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=128,
        help='Batch size for training (default: 128)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    parser.add_argument(
        '--lambdas', type=float, nargs='+',
        default=[1e-5, 1e-4, 1e-3],
        help='List of lambda values to experiment with (default: 1e-5 1e-4 1e-3)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data',
        help='Directory for CIFAR-10 data (default: ./data)'
    )
    parser.add_argument(
        '--results-dir', type=str, default='./results',
        help='Directory for saving results (default: ./results)'
    )
    parser.add_argument(
        '--num-workers', type=int, default=0,
        help='Number of data loading workers (default: 0 for Windows compatibility)'
    )
    return parser.parse_args()


def print_banner():
    """Print a professional banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║       🧠  THE SELF-PRUNING NEURAL NETWORK                       ║
    ║       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        ║
    ║       A network that learns to prune itself during training      ║
    ║       via learnable gate parameters and L1 regularization.       ║
    ║                                                                  ║
    ║       Dataset:  CIFAR-10 (10 classes, 32×32 RGB images)         ║
    ║       Method:   Gated weights + L1 sparsity penalty             ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_experiment(
    lambda_val: float,
    train_loader,
    test_loader,
    args,
    device: str,
    results_dir: str,
) -> dict:
    """
    Run a complete training experiment for a single λ value.

    Args:
        lambda_val: Sparsity coefficient.
        train_loader: Training data loader.
        test_loader: Test data loader.
        args: Parsed arguments.
        device: Device string.
        results_dir: Directory to save results.

    Returns:
        dict: Experiment results including accuracy, sparsity, and metrics.
    """
    print(f"\n{'━'*80}")
    print(f"  EXPERIMENT: λ = {lambda_val}")
    print(f"{'━'*80}")

    # Reset seed for each experiment for fair comparison
    set_seed(args.seed)

    # Create model
    model = SelfPruningNetwork(
        input_dim=3072,
        num_classes=10,
        dropout_rate=0.2,
    )

    print(f"  Model Parameters: {model.get_total_parameters():,}")
    print(f"  Architecture: {[l.in_features for l in model.get_prunable_layers()]} → "
          f"{model.get_prunable_layers()[-1].out_features}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        lambda_sparse=lambda_val,
        learning_rate=args.lr,
        weight_decay=1e-4,
        device=device,
    )

    # Train
    tracker = trainer.train(
        num_epochs=args.epochs,
        log_interval=5,
        save_dir=os.path.join(results_dir, "checkpoints"),
    )

    # Final evaluation
    final_acc = tracker.test_accuracies[-1]
    best_acc = max(tracker.test_accuracies)
    final_sparsity = tracker.sparsity_levels[-1]
    layer_info = model.get_layer_sparsities()

    # Generate per-experiment plots
    lambda_dir = os.path.join(results_dir, f"lambda_{lambda_val:.0e}")
    os.makedirs(lambda_dir, exist_ok=True)

    # Save model state for deferred plotting
    torch.save(model.state_dict(), os.path.join(lambda_dir, "model_state.pt"))

    # Save tracker data for deferred plotting
    import pickle
    with open(os.path.join(lambda_dir, "tracker.pkl"), 'wb') as f:
        pickle.dump({
            'train_losses': tracker.train_losses,
            'cls_losses': tracker.cls_losses,
            'sparsity_losses': tracker.sparsity_losses,
            'test_accuracies': tracker.test_accuracies,
            'sparsity_levels': tracker.sparsity_levels,
            'per_layer_sparsity': tracker.per_layer_sparsity,
            'epoch_times': tracker.epoch_times,
        }, f)

    if HAS_PLOTTING:
        plot_gate_distribution(
            model, lambda_val,
            os.path.join(lambda_dir, "gate_distribution.png"),
        )

        plot_training_curves(
            tracker, lambda_val,
            os.path.join(lambda_dir, "training_curves.png"),
        )

    # Print per-layer sparsity details
    print(f"\n  Per-Layer Sparsity Breakdown:")
    print(f"  {'Layer':<45} {'Sparsity':>10} {'Pruned/Total':>15}")
    print(f"  {'─'*72}")
    for name, sparsity, total, pruned in layer_info:
        print(f"  {name:<45} {sparsity:>9.2f}% {pruned:>6,}/{total:>6,}")

    return {
        'lambda': lambda_val,
        'best_accuracy': best_acc,
        'final_accuracy': final_acc,
        'accuracy': best_acc,
        'sparsity': final_sparsity,
        'layer_info': layer_info,
        'active_params': model.count_active_parameters(),
        'total_params': model.get_total_parameters(),
        'tracker': tracker,
        'model': model,
    }


def generate_results_table(results: dict) -> str:
    """
    Generate a formatted markdown table of results.

    Args:
        results: Dict mapping λ values to result dictionaries.

    Returns:
        str: Markdown-formatted results table.
    """
    lines = []
    lines.append("| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Active Params | Compression Ratio |")
    lines.append("|:----------:|:-----------------:|:------------------:|:-------------:|:-----------------:|")

    for lam in sorted(results.keys()):
        r = results[lam]
        compression = r['total_params'] / max(r['active_params'], 1)
        lines.append(
            f"| {lam:.0e} | {r['accuracy']:.2f} | {r['sparsity']:.2f} | "
            f"{r['active_params']:,} | {compression:.2f}× |"
        )

    return "\n".join(lines)


def main():
    """Main execution function."""
    args = parse_args()
    print_banner()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    print(f"  Configuration:")
    print(f"    Device:      {device}")
    print(f"    Epochs:      {args.epochs}")
    print(f"    Batch Size:  {args.batch_size}")
    print(f"    Learning Rate: {args.lr}")
    print(f"    Lambda Values: {args.lambdas}")
    print(f"    Seed:        {args.seed}")
    print(f"    Results Dir: {results_dir}")
    print()

    set_seed(args.seed)

    # Load CIFAR-10
    print("  Loading CIFAR-10 dataset...")
    train_loader, test_loader = load_cifar10(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"  ✓ Training samples: {len(train_loader.dataset):,}")
    print(f"  ✓ Test samples:     {len(test_loader.dataset):,}")

    # Run experiments for each lambda
    all_results = {}
    total_start = time.time()

    for lambda_val in args.lambdas:
        result = run_experiment(
            lambda_val=lambda_val,
            train_loader=train_loader,
            test_loader=test_loader,
            args=args,
            device=device,
            results_dir=results_dir,
        )
        all_results[lambda_val] = result

    total_time = time.time() - total_start

    if HAS_PLOTTING:
        print(f"\n{'━'*80}")
        print(f"  GENERATING COMPARISON VISUALIZATIONS")
        print(f"{'━'*80}")

        # Prepare results dict without non-serializable items
        plot_results = {
            lam: {
                'accuracy': r['accuracy'],
                'sparsity': r['sparsity'],
                'layer_info': r['layer_info'],
            }
            for lam, r in all_results.items()
        }

        plot_lambda_comparison(
            plot_results,
            os.path.join(results_dir, "lambda_comparison.png"),
        )

        plot_per_layer_sparsity(
            plot_results,
            os.path.join(results_dir, "per_layer_sparsity.png"),
        )
    else:
        print(f"\n  [INFO] Plotting skipped. Run 'python generate_plots.py' to generate plots.")

    # ---------------------------------------------------------------
    # Print final summary
    # ---------------------------------------------------------------
    print(f"\n{'='*80}")
    print(f"  ╔══════════════════════════════════════╗")
    print(f"  ║       EXPERIMENT RESULTS SUMMARY     ║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"{'='*80}")
    print()

    table_str = generate_results_table(all_results)
    print(table_str)
    print()
    print(f"  Total Experiment Time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print()

    # Save results to JSON
    json_results = {
        str(lam): {
            'lambda': lam,
            'best_accuracy': r['accuracy'],
            'sparsity': r['sparsity'],
            'active_params': r['active_params'],
            'total_params': r['total_params'],
            'per_layer_sparsity': [
                {'name': info[0], 'sparsity': info[1],
                 'total': info[2], 'pruned': info[3]}
                for info in r['layer_info']
            ],
        }
        for lam, r in all_results.items()
    }

    results_path = os.path.join(results_dir, "experiment_results.json")
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✓ Results saved to {results_path}")

    # Save markdown table
    table_path = os.path.join(results_dir, "results_table.md")
    with open(table_path, 'w', encoding='utf-8') as f:
        f.write("# Self-Pruning Neural Network — Experiment Results\n\n")
        f.write(table_str)
        f.write(f"\n\n*Trained on CIFAR-10 for {args.epochs} epochs per λ value.*\n")
        f.write(f"*Total experiment time: {total_time:.1f}s*\n")
    print(f"  ✓ Results table saved to {table_path}")

    print(f"\n  All results and visualizations saved to: {os.path.abspath(results_dir)}/")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
