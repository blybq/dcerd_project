"""
反事实图生成。
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def generate_counterfactual(
    x: Tensor,
    edge_index: Tensor,
    mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    if mask.ndim != 1:
        raise ValueError("mask must be 1D")
    if x.shape[0] != mask.shape[0]:
        raise ValueError("x and mask size mismatch")
    x_counter = x * mask.unsqueeze(1)
    if edge_index.numel() == 0:
        return x_counter, edge_index
    keep_src = mask[edge_index[0]]
    keep_dst = mask[edge_index[1]]
    edge_mask = (keep_src * keep_dst).bool()
    filtered = edge_index[:, edge_mask]
    return x_counter, filtered


__all__ = ["generate_counterfactual"]

