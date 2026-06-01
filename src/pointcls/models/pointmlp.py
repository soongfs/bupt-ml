"""PointMLP: Rethinking Network Design and Local Geometry in Point Cloud.

Reference: "Rethinking Network Design and Local Geometry in Point Cloud:
A Simple Residual MLP Framework" (Ma et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcls.data.dataset import farthest_point_sample


def knn_points(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors in 3D space.

    Args:
        x: Tensor of shape (B, 3, N).
        k: Number of neighbors.

    Returns:
        Tensor of shape (B, N, k) containing indices of nearest neighbors.
    """
    B, C, N = x.shape
    # (B, N, N)
    inner = 2 * torch.matmul(x.transpose(2, 1), x)
    xx = (x ** 2).sum(dim=1, keepdim=True)
    pairwise = xx.transpose(2, 1) + xx - inner

    _, idx = pairwise.topk(k=k, dim=-1, largest=False)
    return idx


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather points by indices.

    Args:
        points: (B, C, N)
        idx: (B, M) or (B, M, k)

    Returns:
        Gathered points of shape matching idx expansion.
    """
    B, C, N = points.shape
    idx_shape = idx.shape

    # Flatten idx
    idx = idx.reshape(B, -1)
    batch_indices = torch.arange(B, device=points.device).view(-1, 1) * N
    idx = idx + batch_indices
    idx = idx.view(-1)

    points = points.transpose(2, 1).contiguous()  # (B, N, C)
    gathered = points.view(B * N, -1)[idx, :]  # (B*M*k, C) or (B*M, C)
    gathered = gathered.view(B, *idx_shape[1:], C)
    gathered = gathered.permute(0, 3, 1, 2).contiguous()  # (B, C, M, k) or (B, C, M)

    return gathered


class GeometricAffine(nn.Module):
    """Geometric Affine transformation for local point groups.

    For each local group of points:
    1. Compute centroid
    2. Center points by subtracting centroid
    3. Apply learned affine transform
    4. Add centroid back
    """

    def __init__(self, channels: int, coord_channels: int = 3):
        super().__init__()
        # Two-layer affine: learn scale + bias per channel for each group
        self.affine_alpha = nn.Sequential(
            nn.Conv1d(coord_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )
        self.affine_beta = nn.Sequential(
            nn.Conv1d(coord_channels, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) — features
            centroids: (B, 3, N) — centroid coordinates

        Returns:
            (B, C, N)
        """
        # alpha and beta from centroids (position-dependent)
        alpha = torch.tanh(self.affine_alpha(centroids))  # (B, C, N)
        beta = torch.tanh(self.affine_beta(centroids))    # (B, C, N)

        # Apply affine: alpha * (x - mu) + beta + mu  (but we simplify)
        # Standard geometric affine: y = alpha⊙x + beta
        return alpha * x + beta


class LocalGrouper(nn.Module):
    """FPS downsampling + kNN grouping to form local patches."""

    def __init__(self, npoints: int, k: int):
        """
        Args:
            npoints: Number of points after FPS downsampling.
            k: Number of neighbors for kNN grouping.
        """
        super().__init__()
        self.npoints = npoints
        self.k = k

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Args:
            xyz: (B, 3, N) — point coordinates
            features: (B, C, N) — point features (can be xyz for first layer)

        Returns:
            new_xyz: (B, 3, npoints) — downsampled centroids
            grouped_features: (B, C, npoints, k) — local groups
        """
        B, C, N = features.shape

        # FPS downsampling
        fps_idx = farthest_point_sample(xyz.transpose(2, 1).contiguous(), self.npoints)
        fps_idx = fps_idx.unsqueeze(2).expand(-1, -1, 3)  # (B, npoints, 3)
        new_xyz = torch.gather(xyz.transpose(2, 1), 1, fps_idx).transpose(2, 1)  # (B, 3, npoints)

        # kNN grouping
        knn_idx = knn_points(xyz, self.k)  # (B, N, k)

        # Gather FPS indices' neighbors
        fps_idx_for_gather = fps_idx[:, :, 0].unsqueeze(-1).expand(-1, -1, self.k)  # (B, npoints, k)
        batch_indices = torch.arange(B, device=xyz.device).view(-1, 1, 1)
        # Get neighbor indices for each centroid
        centroid_neighbors = torch.gather(knn_idx, 1, fps_idx_for_gather)  # (B, npoints, k)

        # Gather features
        grouped_features = index_points(features, centroid_neighbors)  # (B, C, npoints, k)

        return new_xyz, grouped_features


class ResMLPBlock(nn.Module):
    """Residual MLP block: 2-layer MLP with residual connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, npoints, k)

        Returns:
            (B, C, npoints, k)
        """
        return x + self.mlp(x)


class PointMLPStage(nn.Module):
    """One stage of PointMLP: LocalGrouper + GeometricAffine + ResMLPBlock."""

    def __init__(self, in_channels: int, out_channels: int, npoints: int, k: int):
        super().__init__()
        self.grouper = LocalGrouper(npoints, k)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ) if in_channels != out_channels else nn.Identity()
        self.geo_affine = GeometricAffine(out_channels, coord_channels=3)
        self.res_mlp = ResMLPBlock(out_channels)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        """
        Args:
            xyz: (B, 3, N)
            features: (B, C, N)

        Returns:
            new_xyz: (B, 3, npoints)
            new_features: (B, Cout, npoints)
        """
        # Group
        new_xyz, grouped_feat = self.grouper(xyz, features)  # (B, C, npoints, k)

        # Project
        grouped_feat = self.proj(grouped_feat)  # (B, Cout, npoints, k)

        # Geometric Affine: apply per-point, using centroid xyz
        B, C, np, k = grouped_feat.shape
        # Need centroids in feature space — just use new_xyz for affine
        # Apply affine to each neighbor point individually
        # For simplicity, apply affine after max-pooling (common simplification)
        # Actually the paper applies it as attention on centroids
        # We'll apply it to the max-pooled features
        max_feat = grouped_feat.max(dim=-1)[0]  # (B, Cout, npoints)
        max_feat = self.geo_affine(max_feat, new_xyz)  # (B, Cout, npoints)

        # Residual MLP on grouped features then max-pool
        grouped_feat = self.res_mlp(grouped_feat)  # (B, Cout, npoints, k)
        pooled = grouped_feat.max(dim=-1)[0]  # (B, Cout, npoints)

        # Combine
        out = pooled + max_feat  # (B, Cout, npoints)

        return new_xyz, out


class PointMLP(nn.Module):
    """PointMLP for point cloud classification."""

    def __init__(
        self,
        num_classes: int = 40,
        use_normals: bool = False,
        elite: bool = True,
        dropout: float = 0.5,
    ):
        super().__init__()
        input_dim = 6 if use_normals else 3

        if elite:
            emb_dims = [128, 256, 512, 1024]
        else:
            emb_dims = [64, 128, 256, 512]

        npoints_list = [512, 256, 128, 64]
        k_list = [24, 24, 12, 12]

        self.stage1 = PointMLPStage(input_dim, emb_dims[0], npoints_list[0], k_list[0])
        self.stage2 = PointMLPStage(emb_dims[0], emb_dims[1], npoints_list[1], k_list[1])
        self.stage3 = PointMLPStage(emb_dims[1], emb_dims[2], npoints_list[2], k_list[2])
        self.stage4 = PointMLPStage(emb_dims[2], emb_dims[3], npoints_list[3], k_list[3])

        # After 4 stages we have emb_dims[3] channels, max+avg pool -> 2*emb_dims[3]
        pool_dim = emb_dims[3] * 2

        self.classifier = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, N) or (B, N, 3)

        Returns:
            logits: (B, num_classes)
        """
        # Ensure (B, 3, N) format
        if x.shape[1] != 3 and x.shape[2] == 3:
            x = x.transpose(2, 1).contiguous()
        elif x.shape[1] == 3:
            pass
        else:
            pass

        xyz = x[:, :3, :]  # Always use first 3 as coordinates
        features = x  # Can be (B, 3, N) or (B, 6, N)

        xyz1, feat1 = self.stage1(xyz, features)
        xyz2, feat2 = self.stage2(xyz1, feat1)
        xyz3, feat3 = self.stage3(xyz2, feat2)
        xyz4, feat4 = self.stage4(xyz3, feat3)

        # Global pooling
        max_pool = feat4.max(dim=2)[0]  # (B, C)
        avg_pool = feat4.mean(dim=2)    # (B, C)
        pooled = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2*C)

        logits = self.classifier(pooled)
        return logits
