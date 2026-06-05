"""DGCNN: Dynamic Graph CNN for point cloud classification.

Reference: "Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019)
"""

import torch
import torch.nn as nn


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Compute k-nearest neighbors in feature space.

    Args:
        x: Tensor of shape (B, C, N).
        k: Number of neighbors.

    Returns:
        Tensor of shape (B, N, k) containing indices of nearest neighbors.
    """
    B, C, N = x.shape
    # Compute pairwise distance matrix
    # (B, N, N) = (B, C, N) -> (B, N, C) @ (B, C, N)
    inner = 2 * torch.matmul(x.transpose(2, 1), x)  # (B, N, N)
    xx = (x ** 2).sum(dim=1, keepdim=True)  # (B, 1, N)
    pairwise = xx.transpose(2, 1) + xx - inner  # (B, N, N)

    _, idx = pairwise.topk(k=k, dim=-1, largest=False)  # (B, N, k)
    return idx


def get_graph_feature(
    x: torch.Tensor,
    k: int = 20,
    coord_dims: int | None = None,
) -> torch.Tensor:
    """Build edge features from k-nearest neighbors.

    Args:
        x: Tensor of shape (B, C, N).
        k: Number of neighbors.
        coord_dims: If set, build the kNN graph using only the first
            coord_dims channels, while gathering edge features from all C
            channels. This is useful for point clouds with normals: xyz should
            define geometry, normals should be attributes.

    Returns:
        Tensor of shape (B, 2*C, N, k) with [central, neighbor-central] features.
    """
    B, C, N = x.shape
    graph_x = x[:, :coord_dims, :] if coord_dims is not None else x
    idx = knn(graph_x, k)  # (B, N, k)

    # Gather neighbor features
    # Expand idx to (B, C, N, k) by repeating across channel dim
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()  # (B, N, C)
    # Flatten batch and point dims
    feature = x.view(B * N, -1)[idx, :]  # (B*N*k, C)
    feature = feature.view(B, N, k, C)  # (B, N, k, C)
    feature = feature.permute(0, 3, 1, 2).contiguous()  # (B, C, N, k)

    # Central features
    x_central = x.view(B, N, 1, C).repeat(1, 1, k, 1)  # (B, N, k, C)
    x_central = x_central.permute(0, 3, 1, 2).contiguous()  # (B, C, N, k)

    # Edge features: [central, neighbor - central]
    edge_feature = torch.cat([x_central, feature - x_central], dim=1)  # (B, 2*C, N, k)

    return edge_feature


class EdgeConv(nn.Module):
    """EdgeConv block: kNN graph -> MLP -> max pool over neighbors."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        k: int = 20,
        graph_coord_dims: int | None = None,
    ):
        super().__init__()
        self.k = k
        self.graph_coord_dims = graph_coord_dims
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, N)

        Returns:
            (B, C_out, N)
        """
        edge_feat = get_graph_feature(x, self.k, coord_dims=self.graph_coord_dims)  # (B, 2*C_in, N, k)
        edge_feat = self.conv1(edge_feat)  # (B, mid, N, k)
        edge_feat = self.conv2(edge_feat)  # (B, C_out, N, k)
        # Max pool over neighbor dimension
        out = edge_feat.max(dim=-1, keepdim=False)[0]  # (B, C_out, N)
        return out


class DGCNN(nn.Module):
    """Dynamic Graph CNN for point cloud classification."""

    def __init__(
        self,
        k: int = 20,
        emb_dims: int = 1024,
        dropout: float = 0.5,
        num_classes: int = 40,
        input_dim: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Four EdgeConv layers. When normals are present, the first dynamic
        # graph is still built from xyz only; normals are gathered as per-point
        # attributes in the edge features.
        first_graph_coord_dims = min(3, input_dim)
        self.edge_conv1 = EdgeConv(input_dim, 64, 64, k=k, graph_coord_dims=first_graph_coord_dims)
        self.edge_conv2 = EdgeConv(64, 64, 64, k=k)
        self.edge_conv3 = EdgeConv(64, 128, 128, k=k)
        self.edge_conv4 = EdgeConv(128, 256, 256, k=k)

        # Total channels after concatenation: 64 + 64 + 128 + 256 = 512
        conv_channels = 512

        self.post_conv = nn.Sequential(
            nn.Conv1d(conv_channels, emb_dims, kernel_size=1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2),
        )

        # Global pooling output: emb_dims * 2 (max + avg)
        self.classifier = nn.Sequential(
            nn.Linear(emb_dims * 2, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, N) or (B, N, C), where C is input_dim (3 xyz or 6 xyz+normals)

        Returns:
            logits: (B, num_classes)
        """
        if x.ndim != 3:
            raise ValueError(f"Expected a 3D tensor (B,C,N) or (B,N,C), got shape {tuple(x.shape)}")

        # Accept both common point cloud layouts. Prefer an explicit match
        # against the configured channel count so (B, N, 6) is handled correctly.
        if x.shape[1] == self.input_dim:
            pass  # Already (B, C, N)
        elif x.shape[2] == self.input_dim:
            x = x.transpose(2, 1).contiguous()  # (B, C, N)
        else:
            raise ValueError(
                f"Expected channel dimension {self.input_dim} in axis 1 or 2, got shape {tuple(x.shape)}"
            )

        x1 = self.edge_conv1(x)
        x2 = self.edge_conv2(x1)
        x3 = self.edge_conv3(x2)
        x4 = self.edge_conv4(x3)

        # Concatenate all edge conv outputs along channel dim
        x_cat = torch.cat([x1, x2, x3, x4], dim=1)  # (B, 512, N)

        x_emb = self.post_conv(x_cat)  # (B, 1024, N)

        # Global pooling
        x_max = x_emb.max(dim=2)[0]  # (B, 1024)
        x_avg = x_emb.mean(dim=2)    # (B, 1024)
        x_pool = torch.cat([x_max, x_avg], dim=1)  # (B, 2048)

        logits = self.classifier(x_pool)  # (B, num_classes)
        return logits
