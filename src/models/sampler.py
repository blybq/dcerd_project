"""
Gumbel Top-K 节点采样器。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GumbelTopKSampler(nn.Module):
    def __init__(self, K: int, tau: float = 1.0):
        super().__init__()
        if K <= 0:
            raise ValueError("K must be positive")
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.K = K
        self.tau = tau

    def forward(self, weights: Tensor, K: int | None = None) -> Tensor:
        if weights.ndim != 1:
            raise ValueError("weights must be 1D tensor")
        target_k = K or self.K
        if target_k <= 0:
            raise ValueError("K must be positive")
        if weights.numel() < target_k:
            raise ValueError("weights length must be >= K")
        gumbel = -torch.log(-torch.log(torch.rand_like(weights).clamp_min(1e-20)))
        scores = (torch.log(weights.clamp_min(1e-20)) + gumbel) / self.tau
        _, top_idx = torch.topk(scores, target_k)
        hard_mask = torch.zeros_like(weights)
        hard_mask[top_idx] = 1.0
        soft_mask = torch.softmax(scores, dim=0)
        return hard_mask.detach() + soft_mask - soft_mask.detach()


__all__ = ["GumbelTopKSampler"]

