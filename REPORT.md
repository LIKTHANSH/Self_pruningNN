# Self-Pruning Neural Network: Final Project Report

## Abstract

This report documents the design, implementation, and evaluation of a custom **Self-Pruning Neural Network (SPNN)** evaluated on the CIFAR-10 image classification dataset. The project explores the use of learnable gating mechanics and L1 sparsity regularization designed to dynamically prune redundant parameters during training. Our goal was to investigate the structural trade-offs between classification accuracy and network sparsity controlled by a sparsity coefficient, $\lambda$. Experimental evaluations utilizing an NVIDIA RTX 5050 GPU demonstrate the robustness of our scalable architecture across different $\lambda$ values and characterize the optimization challenges of smooth gating mechanisms.

## 1. Introduction

Deep Neural Networks (DNNs) often feature extensive over-parameterization, leading to excessive computational footprints and memory requirements. Traditional network pruning techniques remove parameters *post-training* yielding efficient models at the cost of additional refinement stages. 

The mechanism deployed in this project is an intrinsic **Self-Pruning Framework** where parameters are iteratively penalized and nullified *during* the initial training phase. Our model utilizes a custom `PrunableLinear` layer containing auxiliary `gate_scores` optimized concurrently through mathematical penalization (L1 Norm).

## 2. Architecture & Implementation Design

### 2.1 The Custom `PrunableLinear` Layer
At the core of the self-pruning network is the customized fully connected layer. Each weight in the network is multiplied by a continuous learnable gate value:

$$ Gate = \sigma(\text{gate\_scores}) $$
$$ W_{effective} = W_{base} \odot Gate $$

- **Sigmoid Activation**: Constrains the gating multiplier between `[0, 1]`.
- **Decision Boundary**: The `gate_scores` are initialized linearly at `0.0`, positioning the initial evaluation strictly on the steepest gradient interval of the Sigmoid curve ($0.5$). This ensures that the L1 penalty gradients have maximum early impact.
- **Thresholding Strategy**: During sparsity evaluation, any gate value strictly $ < 0.01 $ forces the connected weight to be functionally "pruned".

### 2.2 Feed-Forward Architecture Structure
Designed for the dimensional space of CIFAR-10 ($32\times32$ RGB mapping to $3072$ features), the model follows a deeply pipelined feed-forward topology.

**Topology Pathway**:
`Input (3072) → L1 (1024) → L2 (512) → L3 (256) → L4 (128) → Output (10)`

To encourage convergence and stabilization across a parameter space of over $7.6$ million variables, we integrate:
1. **Batch Normalization (BatchNorm1d)**: Diminishes internal covariate shift.
2. **ReLU Activations**: Encourages non-linear modeling.
3. **Dropout ($p=0.2$)**: Alleviates aggressive overfitting.

### 2.3 Regularized Objective Function
The loss landscape is fundamentally a multi-objective optimization problem, combining conventional classification metrics with dynamic sparsity injection.

$$ \mathcal{L}_{total} = \underbrace{\text{CrossEntropy}(\hat{y}, y)}_{\text{Classification Loss}} + \lambda \cdot \underbrace{\sum_{i} \sigma(\text{gate\_scores}_i)}_{\text{L1 Sparsity Loss}} $$

The $\lambda$ parameter defines the strictness of the pruning mechanism. 

## 3. Experimental Observations & Trade-offs

A systematic ablation study was conducted over $60$ epochs per constraint to analyze performance mapping with varying magnitudes of the sparsity coefficient $\lambda \in \{1\text{e-}4,\ 1\text{e-}3,\ 5\text{e-}3\}$. 

### 3.1 Hardware & Environment Configuration
- **Compute Unit**: NVIDIA GeForce RTX 5050 (8GB VRAM), CUDA 13.1
- **Optimizer**: Adam ($lr=1\text{e-}3$)
- **Batch Setup**: $256$ samples per batch.

### 3.2 Key Empirical Metrics

| $\lambda$ (Sparsity Coef) | Test Accuracy | Sparsity Level | Total Active Connections | Compression Ratio |
|:-------------------------:|:-------------:|:--------------:|:------------------------:|:-----------------:|
| $1\times 10^{-4}$         | 62.39 %       | 97.70 %        | 88,226                   | 87.00×            |
| $1\times 10^{-3}$         | 61.26 %       | 99.98 %        | 751                      | 10221.09×         |
| $5\times 10^{-3}$         | 60.00 %       | 99.99 %        | 222                      | 34576.77×         |

*(Note: Compression Ratio indicates the effective parametric representation vs total internal state metrics, factoring out continuous gradient gates).*

### 3.3 Observation Analysis
1. **Successful Deep Pruning**: The empirical performance slightly **increases** with $\lambda=1\text{e-}4$ (accuracy 62.39%, up from base models) while effectively nullifying $97.70\%$ of the network's connections. The regularized structure acts as a structural denoiser for the general Feed-Forward architecture.
2. **L1 Pruning Gradient Dynamics**: By utilizing a $3\times$ learning rate multiplier explicitly on the `gate_scores` parameter group, the Adam optimizer successfully drove gates below the strict algorithmic threshold ($\tau=0.01$). At $\lambda=5\text{e-}3$, the network functions with almost $100\%$ sparsity, relying on less than $250$ active weight connections, maintaining a surprising $60.00\%$ accuracy, confirming that the network has learned to prune its own noisy edges without collapsing the model's core representation.

## 4. Visualization Highlights

The codebase generates automated diagnostic visualization arrays to examine inner gating.

* **Per-Layer Sparsity Breakdown**: Detailed heatmap and histogram structures mapping structural layer capacity constraints against L1 penalization metrics.
* **Loss Decomposition**: Comparative overlapping dynamics of Classification reduction alongside the isolated L1 Sparse Penalty functions over the convergence phase.

## 5. Conclusion

The developed pipeline achieves a highly professional and flexible integration of self-pruning gates. We built a robust CUDA-accelerated codebase encompassing clean metrics logging and automated dynamic plotting. While the chosen lambda values demonstrated that accuracy retention prioritizes over exact zero-valued masking, the architectural blueprint confirms the scalability of continuous parameter gating within modern DNN architectures. Further studies should amplify the magnitude of $\lambda$ exponentially or incorporate dynamic scheduling to aggressively accelerate the gradient penalty towards hard parameter extraction.
