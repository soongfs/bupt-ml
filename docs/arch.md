# Project Architecture

## Overview

This project implements point cloud classification on the ModelNet40 benchmark using two distinct neural architectures: DGCNN (Dynamic Graph CNN) and PointMLP (Residual MLP Framework). The project follows a modular `src` layout with a unified command-line interface based on Click, enabling seamless switching between models while sharing the data pipeline and training infrastructure.

## Project Structure

```
bupt-ml/
├── pyproject.toml          # Package configuration & dependencies
├── README.md               # Quick-start guide
├── configs/                # YAML hyperparameter configs
│   ├── dgcnn.yaml
│   └── pointmlp.yaml
├── src/pointcls/           # Main package
│   ├── cli.py              # Click-based CLI entry point
│   ├── train.py            # Training loop & evaluation
│   ├── test.py             # Inference with voting
│   ├── data/
│   │   ├── download.py     # Dataset download & extraction
│   │   ├── dataset.py      # ModelNet40 Dataset + FPS
│   │   └── augment.py      # Rotation, scaling, jitter
│   └── models/
│       ├── dgcnn.py        # DGCNN implementation
│       └── pointmlp.py     # PointMLP implementation
├── docs/
│   ├── arch.md             # This document
│   ├── dgcnn.md            # DGCNN details
│   ├── pointmlp.md         # PointMLP details
│   └── design.md           # Chinese design overview
├── scripts/
│   └── run_all.sh          # One-click training script
└── runs/                   # Output: checkpoints & logs
```

## Module Topology

The project follows a linear data flow through clearly separated modules:

1. **download.py** — Fetches ModelNet40 from Princeton's server, extracts, and verifies 40 class directories.

2. **dataset.py** — Parses OFF files, applies Farthest Point Sampling to produce fixed-size point clouds, normalizes to unit sphere, and provides PyTorch DataLoader-compatible batches.

3. **augment.py** — Applies random SO(3) rotations, anisotropic scaling, and Gaussian jitter during training for improved generalization.

4. **models/** — Two model architectures sharing the same input/output contract: (B, 3, N) tensor input, (B, 40) logits output.

5. **train.py** — Orchestrates the training loop: dataset loading, model instantiation, optimizer/scheduler setup, epoch iteration, evaluation, checkpointing, and curve plotting.

6. **test.py** — Implements multi-view voting inference: applies N random rotations to each point cloud, averages logits, and produces CSV predictions.

7. **cli.py** — Exposes all functionality through Click commands: `download`, `train`, `test`, `train_all`, `test_both`.

## Design Decisions

**Why src layout?** The `src` layout isolates the package code from configuration and scripts, preventing accidental imports from the project root. This is the modern Python packaging standard (PEP 517/621) and ensures clean import paths (`from pointcls.models import DGCNN`).

**Why a combined framework?** While DGCNN and PointMLP are architecturally distinct, they share substantial infrastructure: data loading, FPS sampling, augmentation, training loop, evaluation metrics, and checkpoint management. A unified codebase eliminates duplication and ensures consistent experimental conditions for fair comparison.

**Why voting strategy?** Point cloud classification is sensitive to orientation. By averaging predictions over multiple random SO(3) rotations, the voting strategy exploits rotational symmetry and improves accuracy by 1-2 percentage points. This is a standard technique in the point cloud literature.

**Why PyTorch-only FPS?** The project avoids external dependencies like torch-cluster or torch-point3d for Farthest Point Sampling. The pure PyTorch implementation is portable, educational, and eliminates CUDA version compatibility issues.

**Why DataParallel not DDP?** For single-machine multi-GPU training on typical academic hardware, DataParallel offers simpler debugging and configuration without the overhead of distributed process groups. DDP would be appropriate for multi-node clusters.
