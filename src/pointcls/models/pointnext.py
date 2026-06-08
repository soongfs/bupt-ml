"""PointNeXt-style point cloud classifier.

This is a compact PointNeXt-inspired implementation for ModelNet40. It uses
hierarchical FPS downsampling, xyz kNN local aggregation, residual inverted MLP
blocks, and global max+avg pooling. Normals are treated as point features while
all neighborhood geometry is built from xyz only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pointcls.data.dataset import farthest_point_sample
from pointcls.models.pointmlp import knn_points


def index_points_bcn(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather channel-first points/features (B, C, N) by indices (B, S[, K])."""
    B, C, _ = points.shape
    flat_idx = idx.reshape(B, -1)
    gather_idx = flat_idx.unsqueeze(1).expand(-1, C, -1)
    gathered = torch.gather(points, 2, gather_idx)
    if idx.ndim == 2:
        return gathered.view(B, C, idx.shape[1])
    return gathered.view(B, C, idx.shape[1], idx.shape[2])


class ConvBNAct1d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, act: bool = True):
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class ConvBNAct2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, act: bool = True):
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class InvertedResidualMLP(nn.Module):
    """PointNeXt inverted residual MLP block on per-point features."""

    def __init__(self, channels: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.net = nn.Sequential(
            ConvBNAct1d(channels, hidden),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            ConvBNAct1d(hidden, channels, act=False),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class LocalAggregation(nn.Module):
    """Aggregate kNN local neighborhoods using relative xyz and features."""

    def __init__(self, channels: int, nsample: int = 24):
        super().__init__()
        self.nsample = nsample
        # grouped: center feat, neighbor-center feat, relative xyz = 2C + 3
        self.mlp = nn.Sequential(
            ConvBNAct2d(channels * 2 + 3, channels),
            ConvBNAct2d(channels, channels),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        B, C, N = features.shape
        k = min(self.nsample, N)
        idx = knn_points(xyz, k=k)  # (B, N, k), xyz-only graph
        grouped_feat = index_points_bcn(features, idx)  # (B, C, N, k)
        grouped_xyz = index_points_bcn(xyz, idx)  # (B, 3, N, k)
        center_feat = features.unsqueeze(-1).expand(-1, -1, -1, k)
        center_xyz = xyz.unsqueeze(-1).expand(-1, -1, -1, k)
        edge = torch.cat(
            [center_feat, grouped_feat - center_feat, grouped_xyz - center_xyz],
            dim=1,
        )
        return self.mlp(edge).max(dim=-1)[0]


class PointNeXtBlock(nn.Module):
    def __init__(self, channels: int, nsample: int = 24, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        self.local_aggregation = LocalAggregation(channels, nsample=nsample)
        self.mlp = InvertedResidualMLP(channels, expansion=expansion, dropout=dropout)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        features = features + self.local_aggregation(xyz, features)
        return self.mlp(features)


class PointNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        blocks: int,
        nsample: int,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.stride = stride
        self.proj = ConvBNAct1d(in_channels, out_channels)
        self.blocks = nn.ModuleList(
            [PointNeXtBlock(out_channels, nsample=nsample, expansion=expansion, dropout=dropout) for _ in range(blocks)]
        )

    def _downsample(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.stride <= 1:
            return xyz, features
        B, _, N = xyz.shape
        npoints = max(1, N // self.stride)
        idx = farthest_point_sample(xyz.transpose(2, 1).contiguous(), npoints)  # (B, npoints)
        gather_idx = idx.unsqueeze(1).expand(-1, xyz.shape[1], -1)
        new_xyz = torch.gather(xyz, 2, gather_idx)
        gather_feat_idx = idx.unsqueeze(1).expand(-1, features.shape[1], -1)
        new_features = torch.gather(features, 2, gather_feat_idx)
        return new_xyz, new_features

    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz, features = self._downsample(xyz, features)
        features = self.proj(features)
        for block in self.blocks:
            features = block(xyz, features)
        return xyz, features


class PointNeXt(nn.Module):
    """PointNeXt-style classifier for ModelNet40."""

    def __init__(
        self,
        num_classes: int = 40,
        input_dim: int = 3,
        width: int = 64,
        blocks: tuple[int, ...] = (1, 2, 2, 2),
        strides: tuple[int, ...] = (1, 2, 2, 2),
        nsample: int | tuple[int, ...] = 24,
        expansion: int = 4,
        dropout: float = 0.4,
    ):
        super().__init__()
        if len(blocks) != len(strides):
            raise ValueError("blocks and strides must have the same length")
        self.input_dim = input_dim
        nsamples = (nsample,) * len(blocks) if isinstance(nsample, int) else nsample
        if len(nsamples) != len(blocks):
            raise ValueError("nsample must be an int or match blocks length")

        channels = [width * (2 ** i) for i in range(len(blocks))]
        self.stem = nn.Sequential(
            ConvBNAct1d(input_dim, width),
            InvertedResidualMLP(width, expansion=expansion, dropout=dropout * 0.25),
        )
        stages = []
        in_ch = width
        for out_ch, stride, num_blocks, k in zip(channels, strides, blocks, nsamples):
            stages.append(
                PointNeXtStage(
                    in_ch,
                    out_ch,
                    stride=stride,
                    blocks=num_blocks,
                    nsample=k,
                    expansion=expansion,
                    dropout=dropout * 0.25,
                )
            )
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

        final_ch = channels[-1]
        self.head = nn.Sequential(
            nn.Linear(final_ch * 2, final_ch),
            nn.BatchNorm1d(final_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_ch, final_ch // 2),
            nn.BatchNorm1d(final_ch // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(final_ch // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor (B,C,N) or (B,N,C), got {tuple(x.shape)}")
        if x.shape[1] == self.input_dim:
            x = x.contiguous()
        elif x.shape[2] == self.input_dim:
            x = x.transpose(2, 1).contiguous()
        else:
            raise ValueError(
                f"Expected channel dimension {self.input_dim} in axis 1 or 2, got shape {tuple(x.shape)}"
            )

        xyz = x[:, :3, :]
        features = self.stem(x)
        for stage in self.stages:
            xyz, features = stage(xyz, features)
        pooled = torch.cat([features.max(dim=2)[0], features.mean(dim=2)], dim=1)
        return self.head(pooled)
