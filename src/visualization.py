"""
visualization.py
================
Visualization and Plotting Utilities

This module generates publication-quality matplotlib plots for analyzing
the self-pruning network's training dynamics and final state:

    1. Gate Value Distribution — Histogram of all gate values (target: bimodal
       with spike at 0 and cluster near 1)
    2. Sparsity vs. Accuracy Trade-off — How λ affects the balance
    3. Training Curves — Loss and sparsity progression over epochs
    4. Per-Layer Sparsity — Heatmap of pruning across layers

Author: Likthansh Anisetti
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/CI environments

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import os
from typing import Dict, List, Optional

from .network import SelfPruningNetwork
from .trainer import SparsityTracker


# ---------------------------------------------------------------
# Plot styling configuration
# ---------------------------------------------------------------
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

# Color palette
COLORS = {
    'primary': '#58a6ff',
    'secondary': '#7ee787',
    'accent': '#d2a8ff',
    'warning': '#d29922',
    'danger': '#f85149',
    'neutral': '#8b949e',
    'gradient_start': '#1f6feb',
    'gradient_end': '#238636',
}


def plot_gate_distribution(
    model: SelfPruningNetwork,
    lambda_val: float,
    save_path: str,
    threshold: float = 1e-2,
) -> str:
    """
    Plot the distribution of gate values across all PrunableLinear layers.

    A successful self-pruning model should show a bimodal distribution:
        - A large spike at 0 (pruned weights)
        - A cluster of values away from 0 (important weights)

    Args:
        model: Trained SelfPruningNetwork.
        lambda_val: The λ value used during training.
        save_path: File path to save the plot.
        threshold: Gate values below this are considered "pruned".

    Returns:
        str: Path to the saved plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Collect all gate values
    all_gates = []
    layer_gates = {}

    for i, layer in enumerate(model.get_prunable_layers()):
        gate_values = layer.get_gate_values().cpu().numpy().flatten()
        all_gates.extend(gate_values)
        layer_gates[f"Layer {i} ({layer.in_features}→{layer.out_features})"] = gate_values

    all_gates = np.array(all_gates)
    sparsity = model.get_overall_sparsity(threshold)

    # --- Plot 1: Overall Gate Distribution ---
    ax1 = axes[0]
    counts, bins, patches = ax1.hist(
        all_gates, bins=100, range=(0, 1),
        color=COLORS['primary'], alpha=0.85, edgecolor='none',
        density=False,
    )

    # Color pruned region differently
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < threshold:
            patch.set_facecolor(COLORS['danger'])
            patch.set_alpha(0.9)

    ax1.axvline(x=threshold, color=COLORS['warning'], linestyle='--',
                linewidth=1.5, label=f'Prune threshold ({threshold})')
    ax1.set_xlabel('Gate Value (sigmoid output)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Gate Value Distribution (λ={lambda_val})')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add sparsity annotation
    ax1.text(
        0.98, 0.95,
        f'Sparsity: {sparsity:.1f}%\nPruned: {(all_gates < threshold).sum():,}/{len(all_gates):,}',
        transform=ax1.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#21262d', edgecolor=COLORS['primary'], alpha=0.9),
    )

    # --- Plot 2: Per-Layer Gate Distribution (Violin Plot Style) ---
    ax2 = axes[1]
    layer_names = list(layer_gates.keys())
    positions = range(len(layer_names))

    bp = ax2.boxplot(
        [layer_gates[name] for name in layer_names],
        positions=positions,
        vert=True,
        patch_artist=True,
        widths=0.6,
        showfliers=False,
        medianprops=dict(color=COLORS['warning'], linewidth=2),
        whiskerprops=dict(color=COLORS['neutral']),
        capprops=dict(color=COLORS['neutral']),
    )

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
              COLORS['warning'], COLORS['danger']]
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('#c9d1d9')

    ax2.set_xticks(positions)
    ax2.set_xticklabels([f'L{i}' for i in range(len(layer_names))],
                         fontsize=10)
    ax2.set_ylabel('Gate Value')
    ax2.set_title(f'Per-Layer Gate Distribution (λ={lambda_val})')
    ax2.axhline(y=threshold, color=COLORS['danger'], linestyle='--',
                linewidth=1.5, alpha=0.7, label=f'Threshold={threshold}')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  📊 Gate distribution plot saved: {save_path}")
    return save_path


def plot_training_curves(
    tracker: SparsityTracker,
    lambda_val: float,
    save_path: str,
) -> str:
    """
    Plot training curves showing loss decomposition and sparsity progression.

    Four subplots:
        1. Total Loss over epochs
        2. Classification Loss vs. Sparsity Loss
        3. Test Accuracy over epochs
        4. Sparsity Level over epochs

    Args:
        tracker: SparsityTracker with recorded metrics.
        lambda_val: The λ value used.
        save_path: File path to save the plot.

    Returns:
        str: Path to saved plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = range(1, len(tracker.train_losses) + 1)

    # --- Plot 1: Total Loss ---
    ax = axes[0, 0]
    ax.plot(epochs, tracker.train_losses, color=COLORS['primary'],
            linewidth=2, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Loss Decomposition ---
    ax = axes[0, 1]
    ax.plot(epochs, tracker.cls_losses, color=COLORS['secondary'],
            linewidth=2, label='Classification Loss')
    ax.plot(epochs, [l * lambda_val for l in tracker.sparsity_losses],
            color=COLORS['danger'], linewidth=2, alpha=0.8,
            label=f'λ × Sparsity Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Test Accuracy ---
    ax = axes[1, 0]
    ax.plot(epochs, tracker.test_accuracies, color=COLORS['accent'],
            linewidth=2, label='Test Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy')
    best_epoch = np.argmax(tracker.test_accuracies) + 1
    best_acc = max(tracker.test_accuracies)
    ax.axhline(y=best_acc, color=COLORS['warning'], linestyle='--',
               alpha=0.5, label=f'Best: {best_acc:.2f}% (Epoch {best_epoch})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Sparsity Level ---
    ax = axes[1, 1]
    ax.plot(epochs, tracker.sparsity_levels, color=COLORS['warning'],
            linewidth=2, label='Overall Sparsity')
    ax.fill_between(epochs, 0, tracker.sparsity_levels,
                    alpha=0.15, color=COLORS['warning'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Network Sparsity Over Training')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Training Dynamics — λ = {lambda_val}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  📈 Training curves saved: {save_path}")
    return save_path


def plot_lambda_comparison(
    results: Dict,
    save_path: str,
) -> str:
    """
    Plot the sparsity-vs-accuracy trade-off across different λ values.

    Creates a publication-quality dual-axis plot showing how increasing λ
    affects both sparsity (increasing) and accuracy (potentially decreasing).

    Args:
        results: Dict mapping lambda values to (accuracy, sparsity) tuples.
        save_path: File path to save the plot.

    Returns:
        str: Path to saved plot.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    lambdas = sorted(results.keys())
    accuracies = [results[l]['accuracy'] for l in lambdas]
    sparsities = [results[l]['sparsity'] for l in lambdas]
    lambda_labels = [f'{l:.0e}' for l in lambdas]

    # Bar width
    x = np.arange(len(lambdas))
    width = 0.35

    # Accuracy bars
    bars1 = ax1.bar(x - width/2, accuracies, width, label='Test Accuracy (%)',
                    color=COLORS['primary'], alpha=0.85, edgecolor='none')
    ax1.set_xlabel('Lambda (λ) — Sparsity Coefficient', fontsize=13)
    ax1.set_ylabel('Test Accuracy (%)', color=COLORS['primary'], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS['primary'])

    # Sparsity bars on secondary axis
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, sparsities, width, label='Sparsity (%)',
                    color=COLORS['danger'], alpha=0.85, edgecolor='none')
    ax2.set_ylabel('Sparsity Level (%)', color=COLORS['danger'], fontsize=12)
    ax2.tick_params(axis='y', labelcolor=COLORS['danger'])

    # Value labels on bars
    for bar, val in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                 color=COLORS['primary'], fontweight='bold')

    for bar, val in zip(bars2, sparsities):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                 color=COLORS['danger'], fontweight='bold')

    ax1.set_xticks(x)
    ax1.set_xticklabels(lambda_labels, fontsize=11)
    ax1.set_title('Sparsity vs. Accuracy Trade-off Across λ Values',
                  fontsize=15, fontweight='bold', pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center',
               fontsize=11, framealpha=0.8)

    ax1.grid(True, alpha=0.2)
    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  📊 Lambda comparison plot saved: {save_path}")
    return save_path


def plot_per_layer_sparsity(
    results: Dict,
    save_path: str,
) -> str:
    """
    Plot per-layer sparsity as a grouped bar chart for all λ values.

    Shows how different layers in the network respond to pruning pressure,
    revealing which layers are more redundant.

    Args:
        results: Dict mapping lambda values to result dictionaries.
        save_path: File path to save the plot.

    Returns:
        str: Path to saved plot.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    lambdas = sorted(results.keys())
    num_lambdas = len(lambdas)
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
              COLORS['warning'], COLORS['danger']]

    # Get layer names from first result
    first_key = lambdas[0]
    layer_names = [info[0] for info in results[first_key]['layer_info']]
    short_names = [f'L{i}' for i in range(len(layer_names))]
    num_layers = len(layer_names)

    x = np.arange(num_layers)
    width = 0.8 / num_lambdas

    for i, lam in enumerate(lambdas):
        layer_sparsities = [info[1] for info in results[lam]['layer_info']]
        offset = (i - num_lambdas / 2 + 0.5) * width
        bars = ax.bar(x + offset, layer_sparsities, width,
                      label=f'λ={lam:.0e}',
                      color=colors[i % len(colors)],
                      alpha=0.8, edgecolor='none')

    ax.set_xlabel('Layer', fontsize=13)
    ax.set_ylabel('Sparsity (%)', fontsize=12)
    ax.set_title('Per-Layer Sparsity Across λ Values', fontsize=15,
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis='y')

    # Add layer size annotations at bottom
    for i, name in enumerate(layer_names):
        ax.text(i, -7, name.split('(')[1].rstrip(')'),
                ha='center', fontsize=8, color=COLORS['neutral'],
                style='italic')

    plt.tight_layout(pad=2.0)
    fig.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  📊 Per-layer sparsity plot saved: {save_path}")
    return save_path
