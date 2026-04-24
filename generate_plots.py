"""
generate_plots.py
=================
Post-Training Visualization Script

Run this script AFTER training to generate all plots. This uses the saved
model states and tracker data from the results/ directory.

This script is designed to run with the system Python (which has working
matplotlib DLLs), separately from the CUDA-enabled training environment.

Usage:
    python generate_plots.py
    python generate_plots.py --results-dir ./results

Author: Likthansh Anisetti
"""

import os
import sys
import json
import pickle
import argparse

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.network import SelfPruningNetwork
from src.trainer import SparsityTracker
from src.visualization import (
    plot_gate_distribution,
    plot_training_curves,
    plot_lambda_comparison,
    plot_per_layer_sparsity,
)


def main():
    parser = argparse.ArgumentParser(description="Generate plots from training results")
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Directory containing training results')
    args = parser.parse_args()

    results_dir = args.results_dir
    print(f"\n  Generating plots from: {os.path.abspath(results_dir)}")

    # Find all lambda directories
    lambda_dirs = sorted([
        d for d in os.listdir(results_dir)
        if d.startswith('lambda_') and os.path.isdir(os.path.join(results_dir, d))
    ])

    if not lambda_dirs:
        print("  ERROR: No lambda_* directories found. Run train.py first.")
        return

    print(f"  Found {len(lambda_dirs)} experiments: {lambda_dirs}")

    all_results = {}

    for ldir in lambda_dirs:
        lpath = os.path.join(results_dir, ldir)

        # Parse lambda value from directory name
        lam_str = ldir.replace('lambda_', '')
        lambda_val = float(lam_str)

        print(f"\n  Processing {ldir} (lambda={lambda_val})...")

        # Load model state
        model_path = os.path.join(lpath, "model_state.pt")
        tracker_path = os.path.join(lpath, "tracker.pkl")

        if not os.path.exists(model_path):
            print(f"    SKIP: model_state.pt not found")
            continue

        # Create model and load state
        model = SelfPruningNetwork(input_dim=3072, num_classes=10, dropout_rate=0.2)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()

        # Generate gate distribution plot
        plot_gate_distribution(
            model, lambda_val,
            os.path.join(lpath, "gate_distribution.png"),
        )

        # Load and plot tracker data if available
        if os.path.exists(tracker_path):
            with open(tracker_path, 'rb') as f:
                tracker_data = pickle.load(f)

            tracker = SparsityTracker()
            tracker.train_losses = tracker_data['train_losses']
            tracker.cls_losses = tracker_data['cls_losses']
            tracker.sparsity_losses = tracker_data['sparsity_losses']
            tracker.test_accuracies = tracker_data['test_accuracies']
            tracker.sparsity_levels = tracker_data['sparsity_levels']
            tracker.per_layer_sparsity = tracker_data['per_layer_sparsity']
            tracker.epoch_times = tracker_data['epoch_times']

            plot_training_curves(
                tracker, lambda_val,
                os.path.join(lpath, "training_curves.png"),
            )

        # Collect results for comparison plots
        layer_info = model.get_layer_sparsities()
        best_acc = max(tracker.test_accuracies) if os.path.exists(tracker_path) else 0
        sparsity = model.get_overall_sparsity()

        all_results[lambda_val] = {
            'accuracy': best_acc,
            'sparsity': sparsity,
            'layer_info': layer_info,
        }

    # Generate comparison plots
    if len(all_results) >= 2:
        print(f"\n  Generating comparison plots...")

        plot_lambda_comparison(
            all_results,
            os.path.join(results_dir, "lambda_comparison.png"),
        )

        plot_per_layer_sparsity(
            all_results,
            os.path.join(results_dir, "per_layer_sparsity.png"),
        )

    print(f"\n  All plots generated successfully!")
    print(f"  Results directory: {os.path.abspath(results_dir)}")


if __name__ == "__main__":
    main()
