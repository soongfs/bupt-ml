import torch

from pointcls.models import PointNeXt


def test_pointnext_accepts_xyz_and_normals_bnc_and_bcn():
    model = PointNeXt(num_classes=40, input_dim=6, width=16, blocks=(1, 1), strides=(2, 2), nsample=8)
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


def test_pointnext_accepts_xyz_only():
    model = PointNeXt(num_classes=40, input_dim=3, width=16, blocks=(1, 1), strides=(2, 2), nsample=8)
    model.eval()
    points = torch.randn(2, 128, 3)

    with torch.no_grad():
        logits = model(points)

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()
