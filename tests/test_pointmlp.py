import torch

from pointcls.models.pointmlp import LocalGrouper, PointMLP


def test_pointmlp_accepts_xyz_and_normals_bnc_and_bcn():
    model = PointMLP(
        num_classes=40,
        input_dim=6,
        points=128,
        embed_dim=16,
        dim_expansion=(2, 2),
        pre_blocks=(1, 1),
        pos_blocks=(1, 1),
        k_neighbors=(8, 8),
        reducers=(2, 2),
    )
    model.eval()
    points_bnc = torch.randn(2, 128, 6)
    points_bcn = points_bnc.transpose(2, 1).contiguous()

    with torch.no_grad():
        logits_bnc = model(points_bnc)
        logits_bcn = model(points_bcn)

    assert logits_bnc.shape == (2, 40)
    assert logits_bcn.shape == (2, 40)
    assert torch.isfinite(logits_bnc).all()
    assert torch.isfinite(logits_bcn).all()


def test_pointmlp_accepts_xyz_only():
    model = PointMLP(
        num_classes=40,
        input_dim=3,
        points=128,
        embed_dim=16,
        dim_expansion=(2, 2),
        pre_blocks=(1, 1),
        pos_blocks=(1, 1),
        k_neighbors=(8, 8),
        reducers=(2, 2),
    )
    model.eval()
    points = torch.randn(2, 128, 3)

    with torch.no_grad():
        logits = model(points)

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_local_grouper_outputs_pointmlp_full_shape_with_anchor_normalization():
    grouper = LocalGrouper(channel=16, groups=16, kneighbors=8, use_xyz=False, normalize="anchor")
    xyz = torch.randn(2, 64, 3)
    features = torch.randn(2, 64, 16)

    new_xyz, new_features = grouper(xyz, features)

    assert new_xyz.shape == (2, 16, 3)
    assert new_features.shape == (2, 16, 8, 32)
    assert torch.isfinite(new_features).all()


def test_pointmlp_full_default_has_substantive_capacity():
    model = PointMLP(num_classes=40, input_dim=3, points=1024, elite=False)
    params = sum(p.numel() for p in model.parameters())

    assert params > 10_000_000
