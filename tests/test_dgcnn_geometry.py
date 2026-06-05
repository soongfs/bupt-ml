import torch

from pointcls.models.dgcnn import DGCNN, get_graph_feature


def test_dgcnn_accepts_bnc_input_with_normals():
    model = DGCNN(input_dim=6, k=4)
    model.eval()
    points = torch.randn(2, 32, 6)

    with torch.no_grad():
        logits = model(points)

    assert logits.shape == (2, 40)
    assert torch.isfinite(logits).all()


def test_first_edgeconv_can_build_neighbors_from_xyz_only():
    model = DGCNN(input_dim=6, k=2)

    # Same xyz geometry, deliberately different normals. If the first graph is
    # geometric, neighbor indices/features must be identical.
    base = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    normals_a = torch.zeros_like(base)
    normals_b = torch.tensor(
        [
            [100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0],
            [-100.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    xa = torch.cat([base, normals_a], dim=1).unsqueeze(0).transpose(2, 1)
    xb = torch.cat([base, normals_b], dim=1).unsqueeze(0).transpose(2, 1)

    fa = get_graph_feature(xa, k=2, coord_dims=3)
    fb = get_graph_feature(xb, k=2, coord_dims=3)

    # Central xyz and neighbor xyz-difference channels should be identical even
    # when normals differ, proving the graph was constructed from xyz only.
    assert torch.allclose(fa[:, :3], fb[:, :3])
    assert torch.allclose(fa[:, 6:9], fb[:, 6:9])
