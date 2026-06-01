"""Point cloud data augmentation."""

import torch
import numpy as np


def random_rotation_so3(points: torch.Tensor) -> torch.Tensor:
    """Apply a random SO(3) rotation to a point cloud.

    Args:
        points: Tensor of shape (N, 3) or (B, N, 3).

    Returns:
        Rotated points, same shape as input.
    """
    from scipy.spatial.transform import Rotation

    original_shape = points.shape
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (1, N, 3)
        was_2d = True
    else:
        was_2d = False

    B = points.shape[0]
    rotated = []
    for b in range(B):
        R = torch.from_numpy(
            Rotation.random().as_matrix().astype(np.float32)
        )
        # Apply rotation: (N, 3) @ (3, 3) -> (N, 3)
        rotated.append(points[b] @ R.T)

    result = torch.stack(rotated, dim=0)

    if was_2d:
        result = result.squeeze(0)

    return result


def random_scale(points: torch.Tensor, low: float = 0.8, high: float = 1.25) -> torch.Tensor:
    """Apply random scaling to point coordinates.

    Args:
        points: Tensor of shape (..., 3).
        low: Minimum scale factor.
        high: Maximum scale factor.

    Returns:
        Scaled points, same shape as input.
    """
    scale = np.random.uniform(low, high)
    return points * scale


def random_jitter(points: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian jitter to point coordinates.

    Args:
        points: Tensor of shape (..., 3).
        std: Standard deviation of Gaussian noise.

    Returns:
        Jittered points, same shape as input.
    """
    noise = torch.randn_like(points) * std
    return points + noise


def augment_pointcloud(points: torch.Tensor) -> torch.Tensor:
    """Apply augmentation pipeline: rotation -> scaling -> jitter.

    Only operates on xyz coordinates (first 3 dims).

    Args:
        points: Tensor of shape (N, 3) or (N, 6) — only first 3 dims are augmented.

    Returns:
        Augmented points, same shape as input.
    """
    if points.dim() != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) tensor, got {points.shape}")

    xyz = points[:, :3]
    normals = points[:, 3:] if points.shape[1] > 3 else None

    xyz = random_rotation_so3(xyz)
    xyz = random_scale(xyz)
    xyz = random_jitter(xyz)

    if normals is not None:
        points = torch.cat([xyz, normals], dim=1)
    else:
        points = xyz

    return points
