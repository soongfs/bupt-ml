import math

import pytest
import torch

from pointcls.data.augment import augment_pointcloud, random_rotation_z
from pointcls.test import _random_rotate


def _plane_with_normals():
    return torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_random_rotation_z_preserves_height_and_normal_z():
    points = _plane_with_normals()

    rotated = random_rotation_z(points, angle=math.pi / 2)

    expected_xyz = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(rotated[:, :3], expected_xyz, atol=1e-6)
    assert torch.allclose(rotated[:, 2], points[:, 2])
    assert torch.allclose(rotated[:, 5], points[:, 5])


def test_augment_pointcloud_none_does_not_rotate_or_scale_when_disabled():
    points = _plane_with_normals()

    augmented = augment_pointcloud(
        points,
        rotation_mode="none",
        scale=False,
        jitter=False,
    )

    assert torch.equal(augmented, points)


def test_augment_pointcloud_z_keeps_z_coordinates_and_normal_z_when_scale_jitter_disabled():
    points = _plane_with_normals()

    augmented = augment_pointcloud(
        points,
        rotation_mode="z",
        scale=False,
        jitter=False,
    )

    assert torch.allclose(augmented[:, 2], points[:, 2])
    assert torch.allclose(augmented[:, 5], points[:, 5])


def test_augment_pointcloud_rejects_unknown_rotation_mode():
    points = _plane_with_normals()

    with pytest.raises(ValueError, match="rotation_mode"):
        augment_pointcloud(points, rotation_mode="bad-mode")


def test_test_time_rotation_none_returns_clone_without_rotation():
    points = _plane_with_normals()

    rotated = _random_rotate(points, seed=123, rotation_mode="none")

    assert rotated is not points
    assert torch.equal(rotated, points)


def test_test_time_rotation_z_preserves_height_and_normal_z():
    points = _plane_with_normals()

    rotated = _random_rotate(points, seed=123, rotation_mode="z")

    assert torch.allclose(rotated[:, 2], points[:, 2])
    assert torch.allclose(rotated[:, 5], points[:, 5])


def test_test_time_rotation_rejects_unknown_mode():
    points = _plane_with_normals()

    with pytest.raises(ValueError, match="rotation_mode"):
        _random_rotate(points, seed=123, rotation_mode="bad-mode")
