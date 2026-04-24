# Self-Pruning Neural Network — Experiment Results

| Lambda (λ) | Test Accuracy (%) | Sparsity Level (%) | Active Params | Compression Ratio |
|:----------:|:-----------------:|:------------------:|:-------------:|:-----------------:|
| 1e-04 | 62.39 | 97.70 | 88,226 | 87.00× |
| 1e-03 | 61.26 | 99.98 | 751 | 10221.09× |
| 5e-03 | 60.00 | 99.99 | 222 | 34576.77× |

*Trained on CIFAR-10 for 60 epochs per λ value.*
*Total experiment time: 2045.7s*
