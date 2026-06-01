# PointMLP: Residual MLP Framework

## Overview

PointMLP (Ma et al., 2022) challenges the prevailing wisdom that sophisticated geometric operators (convolutions on graphs, continuous kernels, or attention mechanisms) are necessary for point cloud understanding. The paper demonstrates that a carefully designed pure MLP architecture, augmented with a lightweight geometric affine transform, achieves state-of-the-art or competitive performance on ModelNet40, ScanObjectNN, and ShapeNetPart benchmarks.

## Stage Design

PointMLP organizes computation into hierarchical stages, each progressively reducing spatial resolution while increasing feature dimensionality:

| Stage | #Points | Channels (Elite) | Channels (Standard) | kNN |
|-------|---------|------------------|---------------------|-----|
| 1     | 512     | 128              | 64                  | 24  |
| 2     | 256     | 256              | 128                 | 24  |
| 3     | 128     | 512              | 256                 | 12  |
| 4     | 64      | 1024             | 512                 | 12  |

Each stage consists of three sub-modules:

1. **Local Grouper**: Farthest Point Sampling reduces the point count, then k-Nearest Neighbors groups local neighborhoods. The combination of FPS (uniform coverage) and kNN (local structure) provides structured receptive fields.

2. **Geometric Affine**: A learned position-dependent affine transformation applied to each point's features. Given centroid coordinates c and features f:

   ```
   f' = tanh(MLP_α(c)) ⊙ f + tanh(MLP_β(c))
   ```

   where ⊙ denotes element-wise multiplication. The tanh bounds the transformation in [-1, 1], ensuring numerical stability. This transform injects explicit 3D geometric information into the feature space without requiring complex convolution operators.

3. **Residual MLP**: A two-layer shared MLP with batch normalization and ReLU, wrapped in a residual connection. Operating on grouped features (B, C, N, k), the MLP transforms features within each local group, then max-pooling aggregates neighborhood information.

## Elite vs Standard

The project supports two variants:
- **Elite** (default): Uses wider channels [128, 256, 512, 1024], trading parameter count for improved accuracy.
- **Standard**: Uses narrower channels [64, 128, 256, 512], providing a lighter alternative.

## Global Representation

After four stages, the point cloud is reduced to 64 points with 1024 (or 512) feature channels. Global max pooling and global average pooling produce a 2048-dimensional (or 1024-dim) descriptor, which is fed to a two-layer classifier with dropout regularization.

## Key Insight

The success of PointMLP stems from the geometric affine module, which serves as a minimal yet effective bridge between 3D coordinates and feature learning. Unlike DGCNN's explicit edge construction, PointMLP separates geometric processing (affine) from feature transformation (MLP), resulting in a cleaner, more efficient design that achieves comparable accuracy with fewer geometric assumptions.
