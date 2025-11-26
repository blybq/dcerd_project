import torch

from src.models.counterfactual import generate_counterfactual

torch.manual_seed(42)


def test_source_post_preserved():
    x = torch.randn(4, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    mask = torch.tensor([1.0, 0.0, 0.0, 0.0])
    x_cf, edge_cf = generate_counterfactual(x, edge_index, mask)
    assert torch.allclose(x_cf[0], x[0])
    assert edge_cf.shape[1] == 0

