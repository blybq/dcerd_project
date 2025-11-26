import torch

from src.models.dcerd import DCERD

torch.manual_seed(42)


def test_full_forward_backward():
    n_nodes = 50
    n_edges = 100
    x = torch.randn(n_nodes, 64)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    labels = torch.tensor([1])

    model = DCERD(config={"K": 5, "m": 3, "hidden_dim": 64})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    outputs = model(x, edge_index)
    loss = model.compute_loss(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert loss.item() > 0
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)

