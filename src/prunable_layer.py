"""
prunable_layer.py
=================
Custom PrunableLinear Layer Implementation

This module implements a custom linear layer that incorporates learnable "gate"
parameters for each weight. Each gate is a scalar value that, after sigmoid
activation, lies in [0, 1]. When a gate approaches 0, the corresponding weight
is effectively pruned from the network.

The key insight is that by making gates differentiable and learnable, the network
can learn WHICH weights are important during the standard backpropagation process,
rather than relying on post-hoc pruning heuristics.

References:
    - Louizos, C., Welling, M., & Kingma, D. P. (2018). Learning Sparse Neural
      Networks through L0 Regularization. ICLR 2018.
    - Molchanov, D., et al. (2017). Variational Dropout Sparsifies Deep Neural
      Networks. ICML 2017.

Author: Likthansh Anisetti
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PrunableLinear(nn.Module):
    """
    A custom linear layer with learnable gate parameters for self-pruning.

    For each weight w_ij, there is an associated learnable gate score g_ij.
    During the forward pass:
        1. Gate scores are passed through a sigmoid to produce gates ∈ (0, 1).
        2. Pruned weights are computed: pruned_w = w * sigmoid(g).
        3. The output is computed as: y = x @ pruned_w^T + bias.

    This design ensures that gradients flow through both the weight AND the
    gate score parameters, allowing the optimizer to learn which connections
    to keep (gate → 1) and which to prune (gate → 0).

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If True, adds a learnable bias. Default: True.

    Shape:
        - Input: (*, in_features)
        - Output: (*, out_features)

    Attributes:
        weight (torch.nn.Parameter): Learnable weight of shape (out_features, in_features).
        bias (torch.nn.Parameter): Learnable bias of shape (out_features).
        gate_scores (torch.nn.Parameter): Learnable gate scores of shape (out_features, in_features).

    Example:
        >>> layer = PrunableLinear(784, 256)
        >>> x = torch.randn(32, 784)
        >>> output = layer(x)
        >>> output.shape
        torch.Size([32, 256])
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # ---------------------------------------------------------------
        # Standard weight parameter (same as nn.Linear)
        # Initialized with Kaiming Uniform for ReLU compatibility.
        # ---------------------------------------------------------------
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        # ---------------------------------------------------------------
        # Learnable gate scores -- same shape as weight.
        # Initialized to 0.0 so that sigmoid(gate_scores) starts at 0.5.
        # This places each gate at the decision boundary where the sigmoid
        # gradient is maximal (sigma'(0) = 0.25), enabling the optimizer
        # to quickly differentiate important weights (push gate positive)
        # from unimportant ones (push gate negative via L1 penalty).
        # ---------------------------------------------------------------
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        # ---------------------------------------------------------------
        # Optional bias parameter
        # ---------------------------------------------------------------
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize all parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """
        Initialize weight, gate scores, and bias parameters.

        Weight: Kaiming uniform initialization (He initialization), optimal
                for layers followed by ReLU activations.
        Gate scores: Initialized to 0.0, so sigmoid(0.0) = 0.5, placing
                     gates at the decision boundary with maximal gradient.
        Bias: Uniform initialization in [-1/sqrt(fan_in), 1/sqrt(fan_in)].
        """
        # Kaiming uniform for weights (standard PyTorch approach)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Gate scores initialized to 0.0 -> sigmoid(0.0) = 0.5
        nn.init.constant_(self.gate_scores, 0.0)

        # Bias initialization (same as nn.Linear)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gated weights.

        Computation graph:
            gate_scores → sigmoid → gates ∈ (0, 1)
            pruned_weights = weight ⊙ gates    (element-wise multiplication)
            output = x @ pruned_weights^T + bias

        Gradient flow:
            ∂L/∂weight flows through the gate multiplication.
            ∂L/∂gate_scores flows through sigmoid and the weight multiplication.
            Both parameters are updated by the optimizer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Step 1: Transform gate scores to gates via sigmoid → [0, 1]
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiplication to get pruned weights
        pruned_weights = self.weight * gates

        # Step 3: Standard linear transformation: y = xW^T + b
        return F.linear(x, pruned_weights, self.bias)

    def get_gate_values(self) -> torch.Tensor:
        """
        Returns the current gate values (after sigmoid activation).

        Useful for analysis and visualization of the pruning state.

        Returns:
            torch.Tensor: Gate values ∈ (0, 1) with shape (out_features, in_features).
        """
        with torch.no_grad():
            return torch.sigmoid(self.gate_scores)

    def get_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Calculate the sparsity level of this layer.

        Sparsity is defined as the percentage of gates whose value falls
        below the given threshold. A gate below threshold is considered
        "pruned" — its corresponding weight has negligible contribution
        to the network's output.

        Args:
            threshold (float): Gate values below this are considered pruned.
                               Default: 1e-2.

        Returns:
            float: Sparsity percentage ∈ [0, 100].
        """
        gate_values = self.get_gate_values()
        total_gates = gate_values.numel()
        pruned_gates = (gate_values < threshold).sum().item()
        return (pruned_gates / total_gates) * 100.0

    def extra_repr(self) -> str:
        """String representation for printing the layer."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )
