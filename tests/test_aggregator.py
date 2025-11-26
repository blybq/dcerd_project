import torch

from src.models.aggregator import DiversityAggregator

torch.manual_seed(42)


def test_weights_sum_to_one():
    embeddings = torch.randn(3, 64)
    aggregator = DiversityAggregator()
    agg, weights = aggregator(embeddings)
    assert agg.shape[0] == 64
    assert torch.isclose(weights.sum(), torch.tensor(1.0))

