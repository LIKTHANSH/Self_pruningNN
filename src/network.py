"""
network.py
==========
Self-Pruning Feed-Forward Neural Network Architecture

This module defines the SelfPruningNetwork, a multi-layer feed-forward
neural network built entirely with PrunableLinear layers. The network is
designed for image classification on CIFAR-10 (32x32x3 = 3072 input features,
10 output classes).

Architecture Design Choices:
    - Four hidden layers with decreasing width: 1024 -> 512 -> 256 -> 128
    - BatchNorm after each hidden layer for training stability
    - ReLU activations for non-linearity
    - Dropout (0.2) for additional regularization
    - No activation after the final layer (logits for CrossEntropyLoss)

The wider initial layers give the network enough capacity to learn rich
representations before the gate mechanism prunes unnecessary connections.

Author: Likthansh Anisetti
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from .prunable_layer import PrunableLinear


class SelfPruningNetwork(nn.Module):
    """
    A feed-forward neural network using PrunableLinear layers for self-pruning.

    The network learns to classify CIFAR-10 images while simultaneously
    learning which of its own weights are unnecessary via learnable gate
    parameters. An L1 sparsity penalty on the gate values drives the
    network to prune unimportant connections during training.

    Architecture:
        Input (3072) -> PrunableLinear(1024) -> BN -> ReLU -> Dropout
                     -> PrunableLinear(512)  -> BN -> ReLU -> Dropout
                     -> PrunableLinear(256)  -> BN -> ReLU -> Dropout
                     -> PrunableLinear(128)  -> BN -> ReLU -> Dropout
                     -> PrunableLinear(10)   -> Output (logits)

    Args:
        input_dim (int): Dimensionality of input features. Default: 3072 (CIFAR-10).
        num_classes (int): Number of output classes. Default: 10.
        dropout_rate (float): Dropout probability. Default: 0.2.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        num_classes: int = 10,
        dropout_rate: float = 0.2,
    ):
        super(SelfPruningNetwork, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # ---------------------------------------------------------------
        # Network architecture: 5 PrunableLinear layers
        # Hidden dims chosen to provide sufficient capacity for CIFAR-10
        # while being tractable for the pruning mechanism to work with.
        # ---------------------------------------------------------------
        hidden_dims = [1024, 512, 256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                PrunableLinear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
            ])
            prev_dim = hidden_dim

        # Final classification layer (no activation — raw logits)
        layers.append(PrunableLinear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the self-pruning network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Flatten spatial dimensions: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def get_prunable_layers(self) -> List[PrunableLinear]:
        """
        Returns a list of all PrunableLinear layers in the network.

        This is used by the training loop to compute the sparsity loss
        across all gate parameters.

        Returns:
            List[PrunableLinear]: All prunable layers in the network.
        """
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def compute_sparsity_loss(self) -> torch.Tensor:
        """
        Compute the L1 sparsity regularization loss over all gate values.

        The L1 norm (sum of absolute values) of all gate values encourages
        sparsity because it penalizes non-zero gates. Since gates are always
        positive (output of sigmoid), L1 reduces to a simple sum.

        Mathematical formulation:
            SparsityLoss = Sigma_layers Sigma_ij sigmoid(gate_scores_ij)

        This is differentiable with respect to gate_scores, allowing
        the optimizer to push gate scores toward -inf (sigmoid -> 0).

        Returns:
            torch.Tensor: Scalar sparsity loss value.
        """
        sparsity_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for layer in self.get_prunable_layers():
            gate_values = torch.sigmoid(layer.gate_scores)
            sparsity_loss = sparsity_loss + gate_values.sum()

        return sparsity_loss

    def get_overall_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Calculate the overall sparsity level across all prunable layers.

        Args:
            threshold (float): Gate values below this are considered pruned.

        Returns:
            float: Overall sparsity percentage.
        """
        total_gates = 0
        pruned_gates = 0

        for layer in self.get_prunable_layers():
            gate_values = layer.get_gate_values()
            total_gates += gate_values.numel()
            pruned_gates += (gate_values < threshold).sum().item()

        return (pruned_gates / total_gates) * 100.0 if total_gates > 0 else 0.0

    def get_layer_sparsities(self, threshold: float = 1e-2) -> List[Tuple[str, float, int, int]]:
        """
        Get per-layer sparsity information.

        Returns:
            List of tuples: (layer_name, sparsity_%, total_params, pruned_params)
        """
        results = []
        for i, layer in enumerate(self.get_prunable_layers()):
            gate_values = layer.get_gate_values()
            total = gate_values.numel()
            pruned = (gate_values < threshold).sum().item()
            sparsity = (pruned / total) * 100.0
            name = f"PrunableLinear_{i} ({layer.in_features}→{layer.out_features})"
            results.append((name, sparsity, total, pruned))
        return results

    def get_total_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_active_parameters(self, threshold: float = 1e-2) -> int:
        """
        Count the number of weight parameters whose gates are above threshold.

        This represents the effective model size after pruning.

        Args:
            threshold (float): Gate values below this are considered pruned.

        Returns:
            int: Number of active (non-pruned) weight parameters.
        """
        active = 0
        for layer in self.get_prunable_layers():
            gate_values = layer.get_gate_values()
            active += (gate_values >= threshold).sum().item()
        return active
