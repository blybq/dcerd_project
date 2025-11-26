"""
DCE-RD 模型组装。
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
from torch import Tensor, nn

from .aggregator import DiversityAggregator
from .classifier import ClassifierHead
from .counterfactual import generate_counterfactual
from .encoder import GATEncoder
from .sampler import GumbelTopKSampler
from ..losses.total_loss import DCERDLoss


class DCERD(nn.Module):
    def __init__(self, config: Dict | None = None):
        super().__init__()
        cfg = config or {}
        self.hidden_dim = cfg.get("hidden_dim", 64)
        self.num_subgraphs = cfg.get("m", 3)
        self.top_k = cfg.get("K", 15)
        self.tau = cfg.get("tau", 1.0)
        self.encoder = GATEncoder(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            num_layers=3,
        )
        self.sampler = GumbelTopKSampler(K=self.top_k, tau=self.tau)
        self.aggregator = DiversityAggregator()
        self.classifier = ClassifierHead(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim // 2,
            out_dim=2,
        )
        self.loss_fn = DCERDLoss(
            beta=cfg.get("beta", 1.0),
            gamma=cfg.get("gamma", 0.5),
        )

    def forward(self, x: Tensor, edge_index: Tensor) -> Dict[str, Tensor | List[Tensor]]:
        hidden, attention = self.encoder(x, edge_index, return_attention=True)
        node_weights = self._compute_node_weights(attention, hidden)
        sub_embeddings, masks = self._sample_subgraphs(hidden, node_weights)
        aggregated, clf_weights = self.aggregator(sub_embeddings)
        logits = self.classifier(aggregated).unsqueeze(0)
        return {
            "logits": logits,
            "node_embeddings": hidden,
            "node_weights": node_weights,
            "sub_embeddings": sub_embeddings,
            "clf_weights": clf_weights,
            "masks": masks,
            "x": x,
            "edge_index": edge_index,
        }

    def compute_loss(self, outputs: Dict[str, Tensor | List[Tensor]], labels: Tensor) -> Tensor:
        logits = outputs["logits"]  # type: ignore[index]
        masks = outputs["masks"]  # type: ignore[index]
        x = outputs["x"]  # type: ignore[index]
        edge_index = outputs["edge_index"]  # type: ignore[index]

        counter_preds: List[Tensor] = []
        for mask in masks:  # type: ignore[assignment]
            keep_mask = 1.0 - mask
            x_counter, edge_counter = generate_counterfactual(x, edge_index, keep_mask)
            counter_out = self.forward(x_counter, edge_counter)
            counter_preds.append(counter_out["logits"])

        return self.loss_fn(
            y_pred_orig=logits,
            y_true=labels,
            counter_preds=counter_preds,
        )

    def _compute_node_weights(
        self,
        attention_weights,
        embeddings: Tensor,
    ) -> Tensor:
        num_nodes = embeddings.size(0)
        device = embeddings.device
        if attention_weights is None:
            return torch.full((num_nodes,), 1 / max(num_nodes, 1), device=device)
        edge_index, attn = attention_weights
        if attn.dim() > 1:
            attn = attn.mean(dim=1)
        sums = torch.zeros(num_nodes, device=device)
        counts = torch.zeros(num_nodes, device=device)
        sums.scatter_add_(0, edge_index[0], attn)
        counts.scatter_add_(0, edge_index[0], torch.ones_like(attn))
        weights = torch.where(counts > 0, sums / counts, torch.ones_like(sums))
        weights = torch.relu(weights) + 1e-6
        weights = weights / weights.sum().clamp_min(1e-9)
        return weights

    def _sample_subgraphs(
        self,
        embeddings: Tensor,
        node_weights: Tensor,
    ) -> tuple[Tensor, List[Tensor]]:
        num_nodes = embeddings.size(0)
        target_k = min(self.top_k, num_nodes)
        if target_k == 0:
            raise ValueError("Graph must contain at least one node")
        masks: List[Tensor] = []
        sub_vectors: List[Tensor] = []
        for _ in range(self.num_subgraphs):
            mask = self.sampler(node_weights, K=target_k)
            binary_mask = (mask > 0.5).float()
            if binary_mask.sum() == 0:
                binary_mask = torch.ones_like(binary_mask)
            indices = binary_mask.nonzero(as_tuple=False).squeeze(1)
            sub_vectors.append(embeddings[indices].mean(dim=0))
            masks.append(binary_mask)
        return torch.stack(sub_vectors, dim=0), masks


__all__ = ["DCERD"]

