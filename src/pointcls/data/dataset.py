"""ModelNet40 Dataset with FPS sampling and normalization."""

import os
import torch
import numpy as np
from torch.utils.data import Dataset

POINT_EXTENSIONS = (".off", ".txt")


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


def read_txt(filepath: str) -> np.ndarray:
    """Parse a CSV-style ModelNet text point cloud.

    Pointcept's ModelNet40 mirror stores one sample per .txt file with rows:
    x, y, z, nx, ny, nz.

    Args:
        filepath: Path to .txt file.

    Returns:
        Numpy array of shape (N, 3) or (N, 6+).
    """
    try:
        points = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
    except ValueError:
        # Keep a small compatibility fallback for whitespace-delimited files.
        points = np.loadtxt(filepath, dtype=np.float32)

    points = np.atleast_2d(points).astype(np.float32, copy=False)
    if points.shape[1] < 3:
        raise ValueError(f"Expected at least xyz columns in {filepath}, got {points.shape}")
    return points


def read_pointcloud(filepath: str) -> np.ndarray:
    """Read a point cloud from .off or .txt format."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".off":
        return read_off(filepath)
    if ext == ".txt":
        return read_txt(filepath)
    raise ValueError(f"Unsupported point cloud file extension: {filepath}")


class ModelNet40Dataset(Dataset):
    """ModelNet40 point cloud classification dataset.

    Supports split OFF/TXT layout:
        root/
            airplane/
                train/
                    airplane_0001.off
                    ...
                test/
                    airplane_0001.off
                    ...
            ...

    Also supports unsplit Pointcept TXT layout:
        root/
            airplane/
                airplane_0001.txt
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
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        if num_points <= 0:
            raise ValueError(f"num_points must be positive, got {num_points}")
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset root not found: {root}")

        self.root = root
        self.split = split
        self.num_points = num_points
        self.use_normals = use_normals
        self.augment_flag = augment

        # Get sorted class names
        self.classes = sorted(
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
            and not d.startswith("_")
            and not d.startswith(".")
            and d != "__MACOSX"
        )
        if len(self.classes) != 40:
            print(
                f"WARNING: Found {len(self.classes)} class directories in {root}, "
                f"expected 40."
            )
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        # Collect all file paths and labels
        self.filepaths = []
        self.labels = []
        self.layout = "split" if self._has_explicit_split() else "unsplit"

        for cls_name in self.classes:
            if self.layout == "split":
                cls_dir = os.path.join(root, cls_name, split)
                if not os.path.isdir(cls_dir):
                    print(f"WARNING: Directory not found: {cls_dir}")
                    continue
                paths = _list_point_files(cls_dir)
            else:
                cls_dir = os.path.join(root, cls_name)
                paths = _list_point_files(cls_dir)
                split_idx = int(len(paths) * 0.8)
                if split == "train":
                    paths = paths[:split_idx]
                else:
                    paths = paths[split_idx:]

            for path in paths:
                self.filepaths.append(path)
                self.labels.append(self.class_to_idx[cls_name])

        if len(self.filepaths) == 0:
            raise RuntimeError(
                f"No .off or .txt files found for split={split!r} in {root} "
                f"using {self.layout} layout. Check the dataset extraction."
            )

    def _has_explicit_split(self) -> bool:
        return any(
            os.path.isdir(os.path.join(self.root, cls_name, "train"))
            or os.path.isdir(os.path.join(self.root, cls_name, "test"))
            for cls_name in self.classes
        )

    def __len__(self) -> int:
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        filepath = self.filepaths[idx]
        label = self.labels[idx]

        # Read points and keep a stable internal shape: xyz plus optional normals.
        vertices = read_pointcloud(filepath)
        points = torch.from_numpy(vertices.astype(np.float32, copy=False))
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected point cloud with shape (N, >=3), got {points.shape}")
        if points.shape[0] == 0:
            raise ValueError(f"Empty point cloud: {filepath}")
        if points.shape[1] < 6:
            pad = torch.zeros(points.shape[0], 6 - points.shape[1], dtype=points.dtype)
            points = torch.cat([points, pad], dim=1)
        elif points.shape[1] > 6:
            points = points[:, :6]

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


def _list_point_files(directory: str) -> list[str]:
    return [
        os.path.join(directory, fname)
        for fname in sorted(os.listdir(directory))
        if os.path.isfile(os.path.join(directory, fname))
        and fname.lower().endswith(POINT_EXTENSIONS)
    ]
