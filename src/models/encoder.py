"""
GATv2 图编码器（3层堆叠）。

按照开发文档要求实现：
- 3层GATv2编码
- 每层：注意力计算（LeakyReLU）→ 归一化 → 特征聚合 → ReLU激活 → 残差连接
- 返回最后一层的attention权重用于节点重要性计算
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn import GATv2Conv


class GATEncoder(nn.Module):
    """
    3层GATv2编码器，带残差连接。
    
    按照开发文档Layer 2的要求实现：
    - 每层使用GATv2Conv（内部使用LeakyReLU计算attention）
    - 特征聚合后使用ReLU激活
    - 添加残差连接：h_i^(l) = ReLU(h_tilde) + h_i^(l-1)
    - 返回最后一层的attention权重
    
    输入: x [n, d], edge_index [2, E]
    输出: hidden [n, d], attention (可选)
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # 按照文档要求：3层GATv2Conv
        # 第一层：in_dim -> hidden_dim
        # 后续层：hidden_dim -> hidden_dim
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=1, concat=False))
        
        # 后续层：hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False))

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """
        前向传播。
        
        按照文档Layer 2的公式实现：
        1. GATv2Conv内部计算attention（使用LeakyReLU）
        2. 特征聚合
        3. ReLU激活
        4. 残差连接
        
        Args:
            x: 节点特征 [n, in_dim]
            edge_index: 边索引 [2, E]
            return_attention: 是否返回注意力权重
            
        Returns:
            hidden: 编码后的节点特征 [n, hidden_dim]
            attention: 如果 return_attention=True，返回 (edge_index, attention_weights)
                       attention_weights 是每条边的注意力分数 [E]
        """
        h = x
        last_attention = None
        
        for i, conv in enumerate(self.convs):
            # 如果是最后一层且需要返回attention，使用return_attention_weights参数
            if return_attention and i == self.num_layers - 1:
                # PyG的GATv2Conv支持return_attention_weights参数
                # 返回 (output, (edge_index, attention_weights))
                h_new, (edge_idx, attn_weights) = conv(h, edge_index, return_attention_weights=True)
                # attn_weights的形状可能是 [E, heads] 或 [E]，需要处理
                if attn_weights.dim() > 1:
                    # 如果是多头，取平均
                    attn_weights = attn_weights.mean(dim=1)
                last_attention = (edge_idx, attn_weights)
            else:
                # 普通前向传播
                h_new = conv(h, edge_index)
            
            # 按照文档2.4：ReLU激活
            h_new = F.relu(h_new)
            
            # 按照文档2.4：残差连接
            # h_i^(l) = ReLU(h_tilde) + h_i^(l-1)
            # 注意：第一层如果维度不同，不能直接相加
            if i > 0 or h.shape[1] == h_new.shape[1]:
                h = h_new + h
            else:
                # 第一层维度不同时，不使用残差连接
                h = h_new
        
        return h, last_attention if return_attention else None


__all__ = ["GATEncoder"]
