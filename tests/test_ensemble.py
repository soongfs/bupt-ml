import torch
from torch import nn

from pointcls.ensemble import weighted_average_logits


class ConstantModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.logits.unsqueeze(0).repeat(x.shape[0], 1)


def test_weighted_average_logits_combines_model_predictions():
    models = [ConstantModel([2.0, 0.0]), ConstantModel([0.0, 4.0])]
    batch = torch.zeros(3, 3, 8)

    logits = weighted_average_logits(models, batch, weights=[0.75, 0.25])

    assert logits.shape == (3, 2)
    assert torch.allclose(logits[0], torch.tensor([1.5, 1.0]))
