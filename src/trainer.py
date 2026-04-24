"""
trainer.py
==========
Training Engine for the Self-Pruning Neural Network

This module implements the complete training and evaluation pipeline.
It handles:
    - The custom training loop with combined classification + sparsity loss
    - Epoch-level training and evaluation
    - Learning rate scheduling
    - Sparsity tracking and logging throughout training
    - Model checkpointing

The key innovation is the composite loss function:
    Total Loss = CrossEntropyLoss + λ * SparsityLoss

where SparsityLoss = L1(sigmoid(gate_scores)) summed across all layers.

Author: Likthansh Anisetti
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import time
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from .network import SelfPruningNetwork


class SparsityTracker:
    """
    Tracks training metrics across epochs for analysis and visualization.

    Stores per-epoch values of:
        - Training loss (total, classification, sparsity)
        - Test accuracy
        - Overall sparsity level
        - Per-layer sparsity levels
    """

    def __init__(self):
        self.train_losses: List[float] = []
        self.cls_losses: List[float] = []
        self.sparsity_losses: List[float] = []
        self.test_accuracies: List[float] = []
        self.sparsity_levels: List[float] = []
        self.per_layer_sparsity: List[List[float]] = []
        self.epoch_times: List[float] = []

    def log_epoch(
        self,
        train_loss: float,
        cls_loss: float,
        sparsity_loss: float,
        test_acc: float,
        sparsity: float,
        layer_sparsities: List[float],
        epoch_time: float,
    ):
        """Record metrics for one epoch."""
        self.train_losses.append(train_loss)
        self.cls_losses.append(cls_loss)
        self.sparsity_losses.append(sparsity_loss)
        self.test_accuracies.append(test_acc)
        self.sparsity_levels.append(sparsity)
        self.per_layer_sparsity.append(layer_sparsities)
        self.epoch_times.append(epoch_time)


class Trainer:
    """
    Training engine for the Self-Pruning Neural Network.

    Handles the complete training pipeline with the custom loss function
    that combines classification accuracy with sparsity regularization.

    The core loss formulation:
        Total Loss = CE_Loss(predictions, labels) + λ * Σ sigmoid(gate_scores)

    Args:
        model (SelfPruningNetwork): The self-pruning network to train.
        train_loader (DataLoader): Training data loader.
        test_loader (DataLoader): Test data loader.
        lambda_sparse (float): Sparsity regularization coefficient (λ).
        learning_rate (float): Initial learning rate. Default: 1e-3.
        weight_decay (float): L2 weight decay for Adam. Default: 1e-4.
        device (str): Device to train on ('cpu' or 'cuda'). Default: 'cpu'.
    """

    def __init__(
        self,
        model: SelfPruningNetwork,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lambda_sparse: float,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lambda_sparse = lambda_sparse
        self.device = device

        # ---------------------------------------------------------------
        # Classification Loss: Standard Cross-Entropy
        # Combines LogSoftmax and NLLLoss for numerical stability.
        # ---------------------------------------------------------------
        self.criterion = nn.CrossEntropyLoss()

        # ---------------------------------------------------------------
        # Optimizer: Adam with weight decay
        # Adam is chosen for its adaptive learning rates, which work well
        # with the heterogeneous parameter types (weights vs. gate scores).
        # ---------------------------------------------------------------
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # ---------------------------------------------------------------
        # Learning Rate Scheduler: Cosine Annealing
        # Gradually reduces the learning rate following a cosine curve,
        # which has been shown to improve final convergence.
        # ---------------------------------------------------------------
        self.scheduler = None  # Will be set in train()

        # Metrics tracker
        self.tracker = SparsityTracker()

    def _train_epoch(self) -> tuple:
        """
        Train for one epoch.

        For each batch:
            1. Forward pass through the network
            2. Compute classification loss (Cross-Entropy)
            3. Compute sparsity loss (L1 of gate values)
            4. Combine: total_loss = cls_loss + λ * sparsity_loss
            5. Backpropagate and update all parameters (weights + gates)

        Returns:
            tuple: (avg_total_loss, avg_cls_loss, avg_sparsity_loss)
        """
        self.model.train()
        total_loss_sum = 0.0
        cls_loss_sum = 0.0
        sparse_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(images)

            # ---------------------------------------------------------------
            # LOSS COMPUTATION (the core of self-pruning)
            # ---------------------------------------------------------------

            # 1. Classification loss: How well does the network classify?
            cls_loss = self.criterion(logits, labels)

            # 2. Sparsity loss: L1 norm of all gate values
            #    This encourages gates to approach 0, effectively pruning
            #    the corresponding weights from the network.
            sparsity_loss = self.model.compute_sparsity_loss()

            # 3. Total loss: Balance between accuracy and compression
            #    Higher λ → more pruning, potentially lower accuracy
            #    Lower λ  → less pruning, potentially higher accuracy
            total_loss = cls_loss + self.lambda_sparse * sparsity_loss

            # Backpropagation — gradients flow to BOTH weights and gates
            total_loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update parameters
            self.optimizer.step()

            # Accumulate losses
            total_loss_sum += total_loss.item()
            cls_loss_sum += cls_loss.item()
            sparse_loss_sum += sparsity_loss.item()
            num_batches += 1

        return (
            total_loss_sum / num_batches,
            cls_loss_sum / num_batches,
            sparse_loss_sum / num_batches,
        )

    @torch.no_grad()
    def _evaluate(self) -> float:
        """
        Evaluate the model on the test set.

        Returns:
            float: Test accuracy as a percentage.
        """
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in self.test_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return (correct / total) * 100.0

    def train(
        self,
        num_epochs: int = 50,
        log_interval: int = 5,
        save_dir: Optional[str] = None,
    ) -> SparsityTracker:
        """
        Execute the complete training loop.

        Args:
            num_epochs (int): Number of training epochs. Default: 50.
            log_interval (int): Print metrics every N epochs. Default: 5.
            save_dir (Optional[str]): Directory to save model checkpoints.

        Returns:
            SparsityTracker: Object containing all training metrics.
        """
        # Initialize cosine annealing scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )

        print(f"\n{'='*80}")
        print(f"  TRAINING SELF-PRUNING NETWORK")
        print(f"  λ (sparsity coefficient) = {self.lambda_sparse}")
        print(f"  Epochs = {num_epochs}")
        print(f"  Device = {self.device}")
        print(f"  Total Parameters = {self.model.get_total_parameters():,}")
        print(f"{'='*80}\n")

        header = (
            f"{'Epoch':>6} | {'Total Loss':>10} | {'CE Loss':>10} | "
            f"{'Sparse Loss':>11} | {'Test Acc':>9} | {'Sparsity':>9} | {'Time':>6}"
        )
        print(header)
        print("-" * len(header))

        best_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train
            total_loss, cls_loss, sparse_loss = self._train_epoch()

            # Evaluate
            test_acc = self._evaluate()

            # Step learning rate scheduler
            self.scheduler.step()

            # Calculate sparsity
            sparsity = self.model.get_overall_sparsity()
            layer_info = self.model.get_layer_sparsities()
            layer_sparsities = [info[1] for info in layer_info]

            epoch_time = time.time() - epoch_start

            # Log metrics
            self.tracker.log_epoch(
                total_loss, cls_loss, sparse_loss,
                test_acc, sparsity, layer_sparsities, epoch_time
            )

            # Print progress
            if epoch == 1 or epoch % log_interval == 0 or epoch == num_epochs:
                print(
                    f"{epoch:>6} | {total_loss:>10.4f} | {cls_loss:>10.4f} | "
                    f"{sparse_loss:>11.2f} | {test_acc:>8.2f}% | {sparsity:>8.2f}% | "
                    f"{epoch_time:>5.1f}s"
                )

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "test_accuracy": test_acc,
                            "sparsity": sparsity,
                            "lambda_sparse": self.lambda_sparse,
                        },
                        os.path.join(save_dir, f"best_model_lambda_{self.lambda_sparse}.pt"),
                    )

        print(f"\n{'='*80}")
        print(f"  TRAINING COMPLETE")
        print(f"  Best Test Accuracy: {best_acc:.2f}%")
        print(f"  Final Sparsity: {sparsity:.2f}%")
        print(f"  Total Training Time: {sum(self.tracker.epoch_times):.1f}s")
        print(f"{'='*80}\n")

        return self.tracker
