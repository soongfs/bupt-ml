"""Model construction helpers."""

from __future__ import annotations

import torch.nn as nn

from pointcls.models import DGCNN, PointMLP, PointNeXt


def infer_model_name(config: dict, checkpoint_path: str = "") -> str:
    model_name = config.get("model", "")
    if model_name:
        return model_name
    lower = checkpoint_path.lower()
    if "pointnext" in lower:
        return "pointnext"
    if "pointmlp" in lower:
        return "pointmlp"
    if "dgcnn" in lower:
        return "dgcnn"
    raise ValueError("Cannot determine model type from checkpoint or path.")


def build_model(config: dict, checkpoint_path: str = "") -> nn.Module:
    """Instantiate a model from checkpoint/training config."""
    model_name = infer_model_name(config, checkpoint_path)
    use_normals = config.get("use_normals", False)
    if model_name == "dgcnn":
        return DGCNN(
            k=config.get("k", 20),
            emb_dims=config.get("emb_dims", 1024),
            dropout=config.get("dropout", 0.5),
            num_classes=config.get("num_classes", 40),
            input_dim=6 if use_normals else 3,
        )
    if model_name == "pointmlp":
        return PointMLP(
            num_classes=config.get("num_classes", 40),
            input_dim=6 if use_normals else 3,
            points=config.get("num_points", 1024),
            embed_dim=config.get("embed_dim", 64),
            groups=config.get("groups", 1),
            res_expansion=config.get("res_expansion", 1.0),
            activation=config.get("activation", "relu"),
            bias=config.get("bias", False),
            use_xyz=config.get("use_xyz", False),
            normalize=config.get("normalize", "anchor"),
            dim_expansion=tuple(config.get("dim_expansion", [2, 2, 2, 2])),
            pre_blocks=tuple(config.get("pre_blocks", [2, 2, 2, 2])),
            pos_blocks=tuple(config.get("pos_blocks", [2, 2, 2, 2])),
            k_neighbors=tuple(config.get("k_neighbors", [24, 24, 24, 24])),
            reducers=tuple(config.get("reducers", [2, 2, 2, 2])),
            dropout=config.get("dropout", 0.5),
            elite=config.get("elite", None),
        )
    if model_name == "pointnext":
        return PointNeXt(
            num_classes=config.get("num_classes", 40),
            input_dim=6 if use_normals else 3,
            width=config.get("width", 64),
            blocks=tuple(config.get("blocks", [1, 2, 2, 2])),
            strides=tuple(config.get("strides", [1, 2, 2, 2])),
            nsample=tuple(config["nsample"]) if isinstance(config.get("nsample"), list) else config.get("nsample", 24),
            expansion=config.get("expansion", 4),
            dropout=config.get("dropout", 0.4),
        )
    raise ValueError(f"Unknown model: {model_name}")
