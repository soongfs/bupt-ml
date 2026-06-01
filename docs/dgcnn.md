# DGCNN: Dynamic Graph CNN

## Overview

DGCNN (Dynamic Graph CNN), proposed by Wang et al. (2019), introduces EdgeConv, a novel convolution-like operation on point clouds that dynamically constructs a graph in feature space at each layer. Unlike graph CNNs that operate on a fixed graph, EdgeConv recomputes the k-nearest neighbor graph after each layer based on learned feature embeddings, allowing the network to capture non-local semantic relationships beyond Euclidean proximity.

## EdgeConv Mechanism

Given an input point cloud represented as a set of feature vectors X = {x₁, ..., xₙ} where each xᵢ ∈ Rᶜ, EdgeConv computes edge features for each pair (i, jᵢₖ) where jᵢₖ is the k-th nearest neighbor of point i in feature space:

```
eᵢⱼ = h_Θ(xᵢ, xⱼ - xᵢ)
```

where h_Θ is a nonlinear function with learnable parameters Θ. The asymmetric edge function h_Θ(xᵢ, xⱼ - xᵢ) captures both global shape structure (through central point coordinates xᵢ) and local neighborhood information (through the difference xⱼ - xᵢ). The operator is applied with shared MLPs followed by channel-wise max pooling over neighbors:

```
xᵢ' = max_{j∈N(i)} ReLU(θ_m · (xⱼ - xᵢ) + φ_m · xᵢ)
```

In practice, this is implemented as: construct edge features [xᵢ, xⱼ - xᵢ] → Conv2d → BN → LeakyReLU → Conv2d → BN → LeakyReLU → max-pool over k-dim.

## Architecture

```
Input: (B, 3, 1024)
  │
  ├─ EdgeConv(3→64→64, k=20)  ──→ x1 (B, 64, 1024)
  ├─ EdgeConv(64→64→64, k=20) ──→ x2 (B, 64, 1024)
  ├─ EdgeConv(64→128→128, k=20) ──→ x3 (B, 128, 1024)
  ├─ EdgeConv(128→256→256, k=20) ──→ x4 (B, 256, 1024)
  │
  ├─ Concat[x1, x2, x3, x4] → (B, 512, 1024)
  ├─ Conv1d(512→1024) + BN + LeakyReLU
  ├─ Global Max Pool + Global Avg Pool → (B, 2048)
  │
  ├─ FC(2048→512) + BN + LeakyReLU + Dropout(0.5)
  ├─ FC(512→256) + BN + LeakyReLU + Dropout(0.5)
  └─ FC(256→40) → logits
```

## Hyperparameter Rationale

- **k=20**: The number of nearest neighbors balances local detail capture against computational cost. Empirical studies show k=20 works well across most point cloud benchmarks, and increasing k provides diminishing returns.
- **Four EdgeConv layers**: Each layer increases the receptive field in the dynamic graph. The four-layer stack (with increasing channel dimensions 64→64→128→256) progressively abstracts from local geometric features to global shape semantics.
- **Channel concatenation**: Concatenating outputs from all EdgeConv layers (total 512 channels) preserves multi-scale information, combining low-level geometric details from early layers with high-level semantic features from later layers.
- **Dual pooling**: Combining max and average global pooling captures both salient features (max) and overall shape distribution (avg), providing complementary global descriptors.

## Training Strategy

The model uses Adam optimizer with learning rate 0.001, cosine annealing scheduler, and label smoothing (0.2) to prevent overconfidence. Training runs for 250 epochs with batch size 32. Data augmentation includes random SO(3) rotation, anisotropic scaling (0.8-1.25), and Gaussian jitter (σ=0.01).
