"""
最终分类器 MLP。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 64, hidden_dim: int = 32, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)


__all__ = ["ClassifierHead"]

