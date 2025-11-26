import torch

from src.models.sampler import GumbelTopKSampler

torch.manual_seed(42)


def test_mask_has_exactly_k_ones():
    sampler = GumbelTopKSampler(K=5)
    weights = torch.rand(20)
    mask = sampler(weights)
    assert torch.isclose(mask.sum(), torch.tensor(5.0))


def test_gradient_flows_to_weights():
    weights = torch.rand(10, requires_grad=True)
    sampler = GumbelTopKSampler(K=3)
    mask = sampler(weights)
    loss = mask.sum()
    loss.backward()
    assert weights.grad is not None
    assert torch.all(weights.grad != 0)

