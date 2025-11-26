"""
评估指标工具函数。
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

try:
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover - 训练环境应安装
    roc_auc_score = None


def accuracy(logits: Tensor, labels: Tensor) -> float:
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == labels).float().mean()
    return correct.item()


def binary_auc(probs: Tensor, labels: Tensor) -> float:
    if roc_auc_score is None:
        raise RuntimeError("scikit-learn is required for AUC computation")
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    return float(roc_auc_score(labels_np, probs_np))


__all__ = ["accuracy", "binary_auc"]

