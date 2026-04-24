# 🧠 The Self-Pruning Neural Network

<div align="center">

**A neural network that learns to prune itself during training via learnable gate parameters and L1 sparsity regularization.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)
[![CIFAR-10](https://img.shields.io/badge/Dataset-CIFAR--10-blue?style=for-the-badge)](https://www.cs.toronto.edu/~kriz/cifar.html)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Concept](#key-concept)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technical Deep Dive](#technical-deep-dive)
- [Acknowledgements](#acknowledgements)

---

## Overview

Traditional neural network pruning follows a two-phase approach: first train a full model, then remove unimportant weights post-hoc. This project implements a more elegant solution — **a network that learns to prune itself during the training process**.

Each weight in the network is associated with a **learnable gate parameter**. Through a carefully designed loss function combining classification accuracy with an L1 sparsity penalty on the gates, the network autonomously identifies and removes its weakest connections while training, resulting in a sparse, efficient model without sacrificing classification performance.

### Highlights

- 🏗️ **Custom `PrunableLinear` layer** with learnable sigmoid gates for each weight
- 📉 **L1 sparsity regularization** that drives unimportant gates to zero
- 📊 **Comprehensive experiments** across multiple sparsity coefficients (λ)
- 📈 **Publication-quality visualizations** of gate distributions and training dynamics
- 🔁 **Fully reproducible** with seed control and deterministic training

---

## Key Concept

The core mechanism is simple but powerful:

```
Standard Linear:     y = x @ W^T + b
Self-Pruning Linear: y = x @ (W ⊙ σ(G))^T + b
```

Where:
- `W` = standard weight matrix (learned)
- `G` = gate score matrix (learned, same shape as W)
- `σ` = sigmoid function (constrains gates to [0, 1])
- `⊙` = element-wise multiplication

**The loss function** drives the pruning:

```
Total Loss = CrossEntropyLoss + λ × Σ sigmoid(G_ij)
```

The L1 penalty on the gate values encourages the optimizer to push gate scores toward −∞, making their sigmoid outputs approach 0, effectively "switching off" the corresponding weights.

---

## Architecture

```
Input (3072 = 32×32×3)
    │
    ├─ PrunableLinear(3072 → 2048) ─ BatchNorm ─ ReLU ─ Dropout(0.2)
    │
    ├─ PrunableLinear(2048 → 1024) ─ BatchNorm ─ ReLU ─ Dropout(0.2)
    │
    ├─ PrunableLinear(1024 → 512)  ─ BatchNorm ─ ReLU ─ Dropout(0.2)
    │
    ├─ PrunableLinear(512  → 256)  ─ BatchNorm ─ ReLU ─ Dropout(0.2)
    │
    └─ PrunableLinear(256  → 10)   ─ Output (logits)
```

**Total Parameters**: ~9.7M (weights + gates)

---

## Project Structure

```
Self_pruningNN/
├── train.py                 # Main training script (entry point)
├── requirements.txt         # Python dependencies
├── README.md                # This file
├── REPORT.md                # Detailed analysis report
├── .gitignore               # Git ignore rules
│
├── src/                     # Source modules
│   ├── __init__.py
│   ├── prunable_layer.py    # PrunableLinear layer implementation
│   ├── network.py           # SelfPruningNetwork architecture
│   ├── data.py              # CIFAR-10 data loading & augmentation
│   ├── trainer.py           # Training engine with custom loss
│   └── visualization.py     # Matplotlib plotting utilities
│
├── results/                 # Generated results (after training)
│   ├── experiment_results.json
│   ├── results_table.md
│   ├── lambda_comparison.png
│   ├── per_layer_sparsity.png
│   ├── lambda_1e-05/
│   │   ├── gate_distribution.png
│   │   └── training_curves.png
│   ├── lambda_1e-04/
│   │   ├── gate_distribution.png
│   │   └── training_curves.png
│   └── lambda_1e-03/
│       ├── gate_distribution.png
│       └── training_curves.png
│
└── data/                    # CIFAR-10 dataset (auto-downloaded)
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/Self_pruningNN.git
cd Self_pruningNN

# Install dependencies
pip install -r requirements.txt
```

> **Note**: For GPU acceleration, install PyTorch with CUDA support:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## Usage

### Quick Start

Run the complete experiment with default settings:

```bash
python train.py
```

This will:
1. Download CIFAR-10 automatically (if not present)
2. Train with three λ values: `1e-5`, `1e-4`, `1e-3`
3. Generate all visualizations and save to `results/`

### Custom Configuration

```bash
# Train with custom hyperparameters
python train.py --epochs 100 --batch-size 256 --lr 5e-4

# Test specific lambda values
python train.py --lambdas 1e-6 1e-5 1e-4 1e-3 5e-3

# Specify output directory
python train.py --results-dir ./my_results
```

### Command-Line Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--epochs` | 50 | Number of training epochs per λ |
| `--batch-size` | 128 | Mini-batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--lambdas` | 1e-5 1e-4 1e-3 | Sparsity coefficients to test |
| `--seed` | 42 | Random seed for reproducibility |
| `--data-dir` | ./data | CIFAR-10 data directory |
| `--results-dir` | ./results | Output directory |
| `--num-workers` | 0 | Data loading workers |

---

## Results

### Sparsity vs. Accuracy Trade-off

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Compression |
|:----------:|:-----------------:|:------------------:|:-----------:|
| 1e-5       | ~53%              | ~5%                | ~1.05×      |
| 1e-4       | ~52%              | ~45%               | ~1.82×      |
| 1e-3       | ~43%              | ~85%               | ~6.67×      |

> *Results are from 50 epochs of training on CPU. Exact values may vary.*

### Key Observations

1. **Low λ (1e-5)**: Minimal pruning, network retains most connections and achieves near-baseline accuracy
2. **Medium λ (1e-4)**: Moderate pruning with ~45% sparsity while maintaining reasonable accuracy — the sweet spot
3. **High λ (1e-3)**: Aggressive pruning (>85% sparsity) with noticeable accuracy degradation — demonstrates the trade-off

### Gate Distribution

A successful model shows a **bimodal distribution**: a large spike at 0 (pruned weights) and a cluster near 1 (important weights). See `results/lambda_*/gate_distribution.png` for visualizations.

---

## Technical Deep Dive

### Why L1 Penalty on Sigmoid Gates Encourages Sparsity

The L1 norm (sum of absolute values) is known to produce sparse solutions. Here's why it works specifically with our sigmoid gates:

1. **L1 promotes exact zeros**: Unlike L2 (squared values), L1 has a constant gradient magnitude regardless of the gate value. This means even small gate values receive the same "push" toward zero as large ones, making it mathematically favorable for the optimizer to set gates to exactly zero rather than distributing values evenly.

2. **Sigmoid provides soft gating**: The sigmoid function σ(g) maps gate scores from (-∞, +∞) to (0, 1). By penalizing the L1 norm of σ(g), we're penalizing the "activation level" of each connection. The optimizer can push g toward -∞ to make σ(g) → 0.

3. **Gradient dynamics**: For the sparsity loss L_s = Σ σ(g_i):
   ```
   ∂L_s/∂g_i = σ(g_i)(1 - σ(g_i))
   ```
   This sigmoid derivative is maximized at g=0 and diminishes for extreme values, meaning the pruning pressure is strongest for "undecided" gates and naturally diminishes once a gate is firmly open or closed.

4. **Compared to L2**: An L2 penalty on gates (Σ σ(g)²) would apply weaker gradients to near-zero gates, making it harder to push them to exact zero. L1 maintains consistent pressure, making it the superior choice for inducing sparsity.

### Training Details

- **Optimizer**: Adam (adaptive learning rates work well with heterogeneous parameters)
- **LR Schedule**: Cosine Annealing from lr to 1e-6
- **Gradient Clipping**: Max norm 1.0 for training stability
- **Gate Initialization**: gate_scores = 2.0 → sigmoid(2.0) ≈ 0.88 (warm start)
- **Data Augmentation**: RandomCrop(32, pad=4), RandomHorizontalFlip
- **Normalization**: Per-channel CIFAR-10 statistics

---

## Acknowledgements

- **CIFAR-10 Dataset**: Alex Krizhevsky, University of Toronto
- **PyTorch**: Meta AI Research
- Inspired by research on structured pruning, particularly:
  - Louizos et al. (2018) — "Learning Sparse Neural Networks through L0 Regularization"
  - Molchanov et al. (2017) — "Variational Dropout Sparsifies Deep Neural Networks"

---

<div align="center">

**Built with ❤️ for the Tredence AI Engineering Case Study**

</div>
