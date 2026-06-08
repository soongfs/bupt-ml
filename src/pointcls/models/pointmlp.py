"""Full PointMLP implementation for ModelNet40 classification.

This follows the core design of "Rethinking Network Design and Local Geometry in
Point Cloud" (Ma et al., 2022): FPS local grouping, kNN neighborhoods, anchor or
center normalization with learnable affine parameters, pre-extraction residual
MLP blocks on each local group, post-extraction residual MLP blocks on sampled
points, and a global classification head.

The previous in-repo PointMLP was a simplified approximation. This file keeps the
public class name `PointMLP` but implements the full local-group/pre/post block
pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointcls.data.dataset import farthest_point_sample


def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "rrelu":
        return nn.RReLU(inplace=True)
    if name == "selu":
        return nn.SELU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "hardswish":
        return nn.Hardswish(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(inplace=True)
    return nn.ReLU(inplace=True)


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distance.

    Args:
        src: (B, N, C)
        dst: (B, M, C)

    Returns:
        (B, N, M)
    """
    dist = -2 * torch.matmul(src, dst.transpose(2, 1))
    dist = dist + torch.sum(src ** 2, dim=-1, keepdim=True)
    dist = dist + torch.sum(dst ** 2, dim=-1).unsqueeze(1)
    return dist


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather batched points by batched indices.

    Accepts either (B, N, C) point layout (PointMLP) or the legacy (B, C, N)
    layout used by PointNeXt in this project. For 3D indices, returns
    (B, S, K, C) for BNC input and (B, C, S, K) for BCN input.
    """
    if points.ndim != 3:
        raise ValueError("points must be a 3D tensor")

    if idx.numel() == 0:
        raise ValueError("idx must be non-empty")

    idx_max = int(idx.max().item())
    # Legacy project helper behavior: PointNeXt passes features as (B, C, N).
    # If idx cannot index axis 1 but can index axis 2, gather over axis 2 and
    # return channel-first grouped tensors.
    if idx_max >= points.shape[1] and idx_max < points.shape[2]:
        batch_size, channels, num_points = points.shape
        flat_idx = idx.reshape(batch_size, -1)
        gather_idx = flat_idx.unsqueeze(1).expand(-1, channels, -1)
        gathered = torch.gather(points, 2, gather_idx)
        if idx.ndim == 2:
            return gathered.view(batch_size, channels, idx.shape[1])
        return gathered.view(batch_size, channels, idx.shape[1], idx.shape[2])

    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape)
    batch_indices = batch_indices.repeat(repeat_shape)
    return points[batch_indices, idx, :]


def knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN query from new_xyz centers into xyz points.

    Args:
        k: number of neighbors
        xyz: (B, N, 3)
        new_xyz: (B, S, 3)

    Returns:
        (B, S, k) neighbor indices.
    """
    k = min(k, xyz.shape[1])
    dist = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)
    return group_idx


def knn_points(x: torch.Tensor, k: int) -> torch.Tensor:
    """Backward-compatible kNN helper for tensors in (B, C, N) layout."""
    idx = knn_point(k, x.transpose(2, 1).contiguous(), x.transpose(2, 1).contiguous())
    return idx


class ConvBNReLU1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        bias: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvBNReLURes1D(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        hidden_channels = int(channels * res_expansion)
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(hidden_channels),
            get_activation(activation),
            nn.Conv1d(hidden_channels, channels, kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(channels),
        )
        self.act = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + x)


class LocalGrouper(nn.Module):
    """FPS downsampling plus kNN local grouping with PointMLP normalization."""

    def __init__(
        self,
        channel: int,
        groups: int,
        kneighbors: int,
        use_xyz: bool = True,
        normalize: str | None = "center",
    ):
        super().__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.normalize = normalize.lower() if normalize is not None else None
        if self.normalize not in {"center", "anchor", None}:
            raise ValueError("normalize must be one of: center, anchor, None")

        if self.normalize is not None:
            add_channel = 3 if use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones(1, 1, 1, channel + add_channel))
            self.affine_beta = nn.Parameter(torch.zeros(1, 1, 1, channel + add_channel))

    def forward(self, xyz: torch.Tensor, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Group local neighborhoods.

        Args:
            xyz: (B, N, 3)
            points: (B, N, C)

        Returns:
            new_xyz: (B, groups, 3)
            new_points: (B, groups, kneighbors, 2*C + (3 if use_xyz else 0))
        """
        batch_size, num_points, _ = xyz.shape
        groups = min(self.groups, num_points)

        fps_idx = farthest_point_sample(xyz.contiguous(), groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)

        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)

        if self.normalize is not None:
            if self.normalize == "center":
                mean = grouped_points.mean(dim=2, keepdim=True)
            else:
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(2)
            std = torch.std((grouped_points - mean).reshape(batch_size, -1), dim=-1)
            std = std.view(batch_size, 1, 1, 1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        repeated_centers = new_points.unsqueeze(2).repeat(1, 1, idx.shape[-1], 1)
        new_points = torch.cat([grouped_points, repeated_centers], dim=-1)
        return new_xyz, new_points


class PreExtraction(nn.Module):
    """Process each local group independently and max-pool over neighbors."""

    def __init__(
        self,
        channels: int,
        out_channels: int,
        blocks: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
        activation: str = "relu",
        use_xyz: bool = True,
    ):
        super().__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(
                    out_channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, groups, k, d) -> (B, out_channels, groups)
        batch_size, groups, neighbors, channels = x.shape
        x = x.permute(0, 1, 3, 2).reshape(-1, channels, neighbors)
        x = self.transfer(x)
        x = self.operation(x)
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, groups, -1)
        return x.permute(0, 2, 1).contiguous()


class PosExtraction(nn.Module):
    """Residual MLP blocks across sampled points after local pooling."""

    def __init__(
        self,
        channels: int,
        blocks: int = 1,
        groups: int = 1,
        res_expansion: float = 1.0,
        bias: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.operation = nn.Sequential(
            *[
                ConvBNReLURes1D(
                    channels,
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
                for _ in range(blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.operation(x)


def _as_tuple(values: Sequence[int] | int, stages: int, name: str) -> tuple[int, ...]:
    if isinstance(values, int):
        return (values,) * stages
    values = tuple(values)
    if len(values) != stages:
        raise ValueError(f"{name} must have length {stages}, got {len(values)}")
    return values


class PointMLP(nn.Module):
    """Full PointMLP classifier.

    Args:
        input_dim: 3 for xyz only, 6 for xyz+normals. Normals are embedded as
            additional input attributes; xyz remains the geometry used for FPS/kNN.
    """

    def __init__(
        self,
        num_classes: int = 40,
        input_dim: int | None = None,
        use_normals: bool | None = None,
        points: int = 1024,
        embed_dim: int = 64,
        groups: int = 1,
        res_expansion: float = 1.0,
        activation: str = "relu",
        bias: bool = False,
        use_xyz: bool = False,
        normalize: str | None = "anchor",
        dim_expansion: Sequence[int] = (2, 2, 2, 2),
        pre_blocks: Sequence[int] = (2, 2, 2, 2),
        pos_blocks: Sequence[int] = (2, 2, 2, 2),
        k_neighbors: Sequence[int] = (24, 24, 24, 24),
        reducers: Sequence[int] = (2, 2, 2, 2),
        dropout: float = 0.5,
        elite: bool | None = None,
    ):
        super().__init__()
        if input_dim is None:
            input_dim = 6 if use_normals else 3
        if input_dim not in {3, 6}:
            raise ValueError("PointMLP input_dim must be 3 or 6")

        if elite is True:
            embed_dim = 32
            res_expansion = 0.25
            dim_expansion = (2, 2, 2, 1)
            pre_blocks = (1, 1, 2, 1)
            pos_blocks = (1, 1, 2, 1)
        elif elite is False:
            # Full/default PointMLP setting from the reference implementation.
            embed_dim = embed_dim

        stages = len(pre_blocks)
        dim_expansion = _as_tuple(dim_expansion, stages, "dim_expansion")
        pos_blocks = _as_tuple(pos_blocks, stages, "pos_blocks")
        k_neighbors = _as_tuple(k_neighbors, stages, "k_neighbors")
        reducers = _as_tuple(reducers, stages, "reducers")

        self.input_dim = input_dim
        self.stages = stages
        self.points = points
        self.embedding = ConvBNReLU1D(input_dim, embed_dim, bias=bias, activation=activation)

        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()

        last_channel = embed_dim
        anchor_points = points
        for i in range(stages):
            out_channel = last_channel * dim_expansion[i]
            anchor_points = max(anchor_points // reducers[i], 1)
            self.local_grouper_list.append(
                LocalGrouper(last_channel, anchor_points, k_neighbors[i], use_xyz, normalize)
            )
            self.pre_blocks_list.append(
                PreExtraction(
                    last_channel,
                    out_channel,
                    blocks=pre_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                    use_xyz=use_xyz,
                )
            )
            self.pos_blocks_list.append(
                PosExtraction(
                    out_channel,
                    blocks=pos_blocks[i],
                    groups=groups,
                    res_expansion=res_expansion,
                    bias=bias,
                    activation=activation,
                )
            )
            last_channel = out_channel

        act = get_activation(activation)
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _to_bcn(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("PointMLP input must be a 3D tensor")
        if x.shape[1] == self.input_dim:
            return x.contiguous()
        if x.shape[2] == self.input_dim:
            return x.transpose(2, 1).contiguous()
        raise ValueError(
            f"PointMLP expected input_dim={self.input_dim} on axis 1 or 2, got shape {tuple(x.shape)}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_bcn(x)
        xyz = x[:, :3, :].transpose(2, 1).contiguous()  # (B, N, 3)
        features = self.embedding(x)  # (B, D, N)

        for i in range(self.stages):
            xyz, grouped = self.local_grouper_list[i](xyz, features.transpose(2, 1).contiguous())
            features = self.pre_blocks_list[i](grouped)
            features = self.pos_blocks_list[i](features)

        features = F.adaptive_max_pool1d(features, 1).squeeze(-1)
        return self.classifier(features)
