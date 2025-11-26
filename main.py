"""
阶段一：合成数据预演脚本。
"""

from __future__ import annotations

import torch

from src.models.dcerd import DCERD


def _make_dummy_graph(n_nodes: int = 60, n_edges: int = 120):
    x = torch.randn(n_nodes, 64)
    edge_index = torch.randint(0, n_nodes, (2, n_edges))
    label = torch.randint(0, 2, (1,))
    return x, edge_index, label


def run_phase_one(batches: int = 3):
    torch.manual_seed(42)
    model = DCERD(
        config={
            "hidden_dim": 64,
            "K": 10,
            "m": 3,
            "beta": 1.0,
            "gamma": 0.5,
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(batches):
        x, edge_index, label = _make_dummy_graph()
        outputs = model(x, edge_index)
        loss = model.compute_loss(outputs, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("Phase 1 integration test passed!")


if __name__ == "__main__":
    run_phase_one()

