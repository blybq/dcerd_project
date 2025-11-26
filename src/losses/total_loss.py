"""
总损失组合：原始预测 + 反事实预测。
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DCERDLoss(nn.Module):
    def __init__(self, beta: float = 1.0, gamma: float = 0.5):
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        y_pred_orig: Tensor,
        y_true: Tensor,
        counter_preds: Sequence[Tensor],
    ) -> Tensor:
        loss = self.beta * F.cross_entropy(y_pred_orig, y_true)
        if counter_preds:
            flipped = 1 - y_true
            counter_loss = torch.stack(
                [F.cross_entropy(pred, flipped) for pred in counter_preds]
            )
            loss = loss + self.gamma * counter_loss.mean()
        return loss


__all__ = ["DCERDLoss"]

