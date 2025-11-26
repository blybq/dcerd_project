"""
多样性权重计算与聚合模块。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DiversityAggregator(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, sub_embeddings: Tensor) -> tuple[Tensor, Tensor]:
        if sub_embeddings.ndim != 2:
            raise ValueError("sub_embeddings must be [m, d]")
        m = sub_embeddings.shape[0]
        if m == 0:
            raise ValueError("sub_embeddings must contain at least one vector")
        dists = torch.cdist(sub_embeddings, sub_embeddings, p=2)
        mask = ~torch.eye(m, dtype=torch.bool, device=sub_embeddings.device)
        safe_dists = torch.where(mask, dists, torch.ones_like(dists))
        log_terms = torch.log(safe_dists + self.eps) * mask
        scores = log_terms.sum(dim=1)
        weights = torch.softmax(scores, dim=0)
        aggregated = torch.sum(weights.unsqueeze(1) * sub_embeddings, dim=0)
        return aggregated, weights


__all__ = ["DiversityAggregator"]

