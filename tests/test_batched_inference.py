import torch
from torch import nn

from pointcls.test import predict_logits_batched


class SumModel(nn.Module):
    def forward(self, x):
        # x: (B, C, N). Class 0 logit is max radius; class 1 is negative max radius.
        score = (x[:, :3] ** 2).sum(dim=1).max(dim=1)[0]
        return torch.stack([score, -score], dim=1)


def test_predict_logits_batched_returns_one_logit_per_sample():
    samples = [
        torch.tensor([[float(i), 0.0, 0.0] for i in range(8)]),
        -torch.tensor([[float(i), 0.0, 0.0] for i in range(8)]),
        torch.tensor([[float(i % 2), 0.0, 0.0] for i in range(8)]),
    ]
    model = SumModel()

    logits = predict_logits_batched(
        model,
        samples,
        use_normals=False,
        num_votes=1,
        rotation_mode="none",
        batch_size=2,
    )

    assert logits.shape == (3, 2)
    assert logits[0, 0] > logits[0, 1]
    assert logits[1, 0] > logits[1, 1]
    assert logits[2, 0] > logits[2, 1]
