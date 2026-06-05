"""Point cloud data augmentation."""

import math

import numpy as np
import torch

ROTATION_MODES = {"none", "z", "so3"}


def _validate_rotation_mode(rotation_mode: str) -> str:
    if rotation_mode not in ROTATION_MODES:
        raise ValueError(
            f"rotation_mode must be one of {sorted(ROTATION_MODES)}, got {rotation_mode!r}"
        )
    return rotation_mode


def _random_so3_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    from scipy.spatial.transform import Rotation

    return torch.as_tensor(
        Rotation.random().as_matrix().astype(np.float32),
        device=device,
        dtype=dtype,
    )


def _z_rotation_matrix(
    angle: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    c = math.cos(angle)
    s = math.sin(angle)
    return torch.tensor(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
        dtype=dtype,
    )


def _random_z_matrix(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    angle = float(np.random.uniform(0.0, 2.0 * math.pi))
    return _z_rotation_matrix(angle, device, dtype)


def random_rotation_so3(points: torch.Tensor) -> torch.Tensor:
    """Apply a random SO(3) rotation to xyz tensors.

    Args:
        points: Tensor of shape (N, 3) or (B, N, 3).

    Returns:
        Rotated points, same shape as input.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)  # (1, N, 3)
        was_2d = True
    else:
        was_2d = False

    rotated = []
    for b in range(points.shape[0]):
        R = _random_so3_matrix(points.device, points.dtype)
        rotated.append(points[b] @ R.T)

    result = torch.stack(rotated, dim=0)
    return result.squeeze(0) if was_2d else result


def random_rotation_z(points: torch.Tensor, angle: float | None = None) -> torch.Tensor:
    """Apply a z-axis rotation to a point cloud.

    Rotates xyz and normals together for tensors with at least 6 channels.
    Extra channels beyond xyz+normals are preserved unchanged.

    Args:
        points: Tensor of shape (N, 3+) or (B, N, 3+).
        angle: Optional angle in radians. If omitted, sampled uniformly from [0, 2pi).

    Returns:
        Rotated points, same shape as input.
    """
    if points.dim() == 2:
        batched = points.unsqueeze(0)
        was_2d = True
    elif points.dim() == 3:
        batched = points
        was_2d = False
    else:
        raise ValueError(f"Expected (N,C) or (B,N,C) tensor, got {points.shape}")
    if batched.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 channels, got {batched.shape}")

    out = batched.clone()
    for b in range(batched.shape[0]):
        theta = float(np.random.uniform(0.0, 2.0 * math.pi)) if angle is None else angle
        R = _z_rotation_matrix(theta, points.device, points.dtype)
        out[b, :, :3] = batched[b, :, :3] @ R.T
        if batched.shape[-1] >= 6:
            out[b, :, 3:6] = batched[b, :, 3:6] @ R.T
    return out.squeeze(0) if was_2d else out


def random_scale(points: torch.Tensor, low: float = 0.8, high: float = 1.25) -> torch.Tensor:
    """Apply random scaling to point coordinates."""
    scale_factor = np.random.uniform(low, high)
    return points * scale_factor


def random_jitter(points: torch.Tensor, std: float = 0.01) -> torch.Tensor:
    """Add Gaussian jitter to point coordinates."""
    noise = torch.randn_like(points) * std
    return points + noise


def augment_pointcloud(
    points: torch.Tensor,
    rotation_mode: str = "z",
    scale: bool = True,
    jitter: bool = True,
) -> torch.Tensor:
    """Apply augmentation pipeline: rotation -> scaling -> jitter.

    Rotation modes:
        none: no rotation.
        z: rotate around vertical z-axis only, preserving canonical up/down.
        so3: full random SO(3) rotation; experimental for ModelNet40.

    Rotates xyz and normals together; scale/jitter only affect xyz.

    Args:
        points: Tensor of shape (N, 3) or (N, 6+).
        rotation_mode: One of none, z, so3.
        scale: Whether to apply random xyz scaling.
        jitter: Whether to apply random xyz jitter.

    Returns:
        Augmented points, same shape as input.
    """
    rotation_mode = _validate_rotation_mode(rotation_mode)
    if points.dim() != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected (N, >=3) tensor, got {points.shape}")

    out = points.clone()
    if rotation_mode == "none":
        pass
    elif rotation_mode == "z":
        out = random_rotation_z(out)
    else:
        R = _random_so3_matrix(points.device, points.dtype)
        out[:, :3] = out[:, :3] @ R.T
        if out.shape[1] >= 6:
            out[:, 3:6] = out[:, 3:6] @ R.T

    xyz = out[:, :3]
    if scale:
        xyz = random_scale(xyz)
    if jitter:
        xyz = random_jitter(xyz)
    out[:, :3] = xyz
    return out
