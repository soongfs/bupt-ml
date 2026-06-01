"""ModelNet40 Dataset with FPS sampling and normalization."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset


def farthest_point_sample(points: torch.Tensor, npoints: int) -> torch.Tensor:
    """Farthest Point Sampling (FPS) implemented in pure PyTorch.

    Algorithm:
    1. Pick a random starting point.
    2. Iteratively select the point farthest from all currently selected points.

    Args:
        points: Tensor of shape (B, N, 3) — batch of point clouds.
        npoints: Number of points to sample.

    Returns:
        Tensor of shape (B, npoints) containing indices of sampled points.
    """
    B, N, _ = points.shape
    device = points.device

    # Store indices of selected points
    centroids = torch.zeros(B, npoints, dtype=torch.long, device=device)

    # Distance from each point to its nearest selected point
    distances = torch.full((B, N), float("inf"), device=device)

    # Pick a random starting point for each batch
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)

    # Batch indices for gathering
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoints):
        centroids[:, i] = farthest

        # Get coordinates of the farthest point
        centroid_coords = points[batch_indices, farthest, :].unsqueeze(1)  # (B, 1, 3)

        # Compute squared distance from all points to this centroid
        dist = torch.sum((points - centroid_coords) ** 2, dim=2)  # (B, N)

        # Update distances: for each point, take the min of old distance and new distance
        distances = torch.min(distances, dist)

        # Select new farthest point
        farthest = torch.argmax(distances, dim=1)  # (B,)

    return centroids


def normalize_pointcloud(points: torch.Tensor) -> torch.Tensor:
    """Center and scale a point cloud to unit sphere.

    1. Subtract centroid (mean over points).
    2. Divide by max L2 distance from origin.

    Args:
        points: Tensor of shape (N, 3).

    Returns:
        Normalized point cloud, same shape.
    """
    centroid = points.mean(dim=0, keepdim=True)  # (1, 3)
    points = points - centroid
    max_dist = torch.max(torch.norm(points, p=2, dim=1))  # scalar
    if max_dist > 0:
        points = points / max_dist
    return points


def read_off(filepath: str) -> np.ndarray:
    """Parse an OFF file and extract vertices.

    Args:
        filepath: Path to .off file.

    Returns:
        Numpy array of shape (N, 6) with (x, y, z, nx, ny, nz).
    """
    with open(filepath, "r") as f:
        header = f.readline().strip()
        if header != "OFF":
            # Some files have "OFF\r\n" — handle that
            if header.startswith("OFF"):
                pass
            else:
                raise ValueError(f"Not a valid OFF file: {filepath}, header={header}")

        # Read counts line (skip comments)
        line = f.readline().strip()
        while line.startswith("#") or line == "":
            line = f.readline().strip()
        parts = line.split()
        num_vertices = int(parts[0])
        num_faces = int(parts[1])

        vertices = []
        for _ in range(num_vertices):
            vals = f.readline().strip().split()
            vertices.append([float(v) for v in vals])

    return np.array(vertices, dtype=np.float32)


class ModelNet40Dataset(Dataset):
    """ModelNet40 point cloud classification dataset.

    Expects directory structure:
        root/
            airplane/
                train/
                    airplane_0001.off
                    ...
                test/
                    airplane_0001.off
                    ...
            ...
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_points: int = 1024,
        use_normals: bool = False,
        augment: bool = False,
    ):
        """
        Args:
            root: Path to extracted ModelNet40 directory (contains class subdirs).
            split: "train" or "test".
            num_points: Number of points to sample via FPS.
            use_normals: Whether to include normal vectors (6 dims) or just xyz (3 dims).
            augment: Whether to apply data augmentation.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment_flag = augment

        # Get sorted class names
        self.classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        if len(self.classes) != 40:
            print(
                f"WARNING: Found {len(self.classes)} class directories in {root}, "
                f"expected 40."
            )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        self.filepaths = []
        self.labels = []

        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name, split)
            if not os.path.isdir(cls_dir):
                print(f"WARNING: Directory not found: {cls_dir}")
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.endswith(".off"):
                    self.filepaths.append(os.path.join(cls_dir, fname))
                    self.labels.append(self.class_to_idx[cls_name])

        if len(self.filepaths) == 0:
            raise RuntimeError(
                f"No .off files found in {root}/*/ {split}/. "
                f"Check the dataset extraction."
            )

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        # Read points (N, 6): xyz + normals
        vertices = read_off(filepath)  # (N, 6)
        points = torch.from_numpy(vertices)  # (N, 6)

        # FPS sampling
        if points.shape[0] >= self.num_points:
            points_batch = points.unsqueeze(0)  # (1, N, 6)
            xyz = points_batch[:, :, :3]  # (1, N, 3)
            fps_idx = farthest_point_sample(xyz, self.num_points)  # (1, npoints)
            fps_idx = fps_idx.squeeze(0)  # (npoints,)
            points = points[fps_idx]  # (npoints, 6)
        else:
            # If not enough points, pad by repeating
            repeat = self.num_points // points.shape[0] + 1
            points = points.repeat(repeat, 1)[:self.num_points]

        # Normalize xyz
        points[:, :3] = normalize_pointcloud(points[:, :3])

        # Augmentation (on xyz only)
        if self.augment_flag:
            from pointcls.data.augment import augment_pointcloud
            points = augment_pointcloud(points)

        # Select features
        if self.use_normals:
            out = points  # (npoints, 6)
        else:
            out = points[:, :3]  # (npoints, 3)

        return out, label
