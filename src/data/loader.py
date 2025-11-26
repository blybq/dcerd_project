"""
MC-Fake数据集加载器。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Optional

import torch
from torch_geometric.data import Data, Dataset

# 增加CSV字段大小限制以处理大字段（如text、retweet_relations等）
# 默认限制是131072字节，对于包含大量关系数据的CSV可能不够
# 设置为系统最大值或一个很大的值（如10MB）
try:
    # 尝试设置为系统最大值
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    # 如果系统最大值太大，设置为一个合理的大值（10MB）
    csv.field_size_limit(10 * 1024 * 1024)


def build_graph_from_event(
    news_id: str,
    title: str,
    url: str,
    publish_date: str,
    source: str,
    text: str,
    labels: str,
    n_tweets: str,
    n_retweets: str,
    n_replies: str,
    n_users: str,
    tweet_ids: str,
    retweet_ids: str,
    reply_ids: str,
    user_ids: str,
    retweet_relations: str,
    reply_relations: str,
    data_name: str,
) -> Optional[Data]:
    """
    从CSV行构建PyG Data对象。
    
    Args:
        news_id: 事件ID
        title: 新闻标题
        url: 新闻URL
        publish_date: 发布日期
        source: 新闻源
        text: 新闻正文
        labels: 标签 "true"/"false" 或 "1"/"0"
        n_tweets: 推文总数（字符串）
        n_retweets: 转发数（字符串）
        n_replies: 回复数（字符串）
        n_users: 用户数（字符串）
        tweet_ids: 推文ID列表，逗号分隔
        retweet_ids: 转发ID列表，逗号分隔
        reply_ids: 回复ID列表，逗号分隔
        user_ids: 用户ID列表，逗号分隔
        retweet_relations: 转发关系，格式 "tweetA-tweetB-userA-userB," 多个用逗号分隔
        reply_relations: 回复关系，格式同上
        data_name: 数据类别
        
    Returns:
        Data对象，包含 x, edge_index, y；如果n_tweets < 2则返回None
    """
    # 解析n_tweets并过滤
    try:
        n_tweets_int = int(n_tweets.strip())
    except (ValueError, AttributeError):
        n_tweets_int = 0
    
    if n_tweets_int < 2:
        return None
    
    # 解析标签：支持 "true"/"false" 或 "1"/"0"
    labels_lower = labels.strip().lower()
    if labels_lower in ("true", "1"):
        y = torch.tensor([1], dtype=torch.long)
    elif labels_lower in ("false", "0"):
        y = torch.tensor([0], dtype=torch.long)
    else:
        # 默认处理：尝试转换为整数
        try:
            y = torch.tensor([int(labels.strip())], dtype=torch.long)
        except (ValueError, AttributeError):
            y = torch.tensor([0], dtype=torch.long)
    
    # 解析tweet_ids为节点列表
    if not tweet_ids or not tweet_ids.strip():
        return None
    
    tweet_id_list = [tid.strip() for tid in tweet_ids.split(",") if tid.strip()]
    if len(tweet_id_list) < 2:
        return None
    
    # 创建节点ID到索引的映射
    node_id_to_idx = {tid: idx for idx, tid in enumerate(tweet_id_list)}
    num_nodes = len(tweet_id_list)
    
    # 解析边关系
    edges = []
    
    # 解析转发关系：格式 "tweetA-tweetB-userA-userB"
    if retweet_relations and retweet_relations.strip():
        for rel_str in retweet_relations.split(","):
            rel_str = rel_str.strip()
            if not rel_str:
                continue
            parts = rel_str.split("-")
            if len(parts) >= 2:
                tweet_src = parts[0].strip()
                tweet_dst = parts[1].strip()
                # 只使用tweet ID，忽略user ID
                if tweet_src in node_id_to_idx and tweet_dst in node_id_to_idx:
                    src_idx = node_id_to_idx[tweet_src]
                    dst_idx = node_id_to_idx[tweet_dst]
                    edges.append((src_idx, dst_idx))
    
    # 解析回复关系：格式同上
    if reply_relations and reply_relations.strip():
        for rel_str in reply_relations.split(","):
            rel_str = rel_str.strip()
            if not rel_str:
                continue
            parts = rel_str.split("-")
            if len(parts) >= 2:
                tweet_src = parts[0].strip()
                tweet_dst = parts[1].strip()
                if tweet_src in node_id_to_idx and tweet_dst in node_id_to_idx:
                    src_idx = node_id_to_idx[tweet_src]
                    dst_idx = node_id_to_idx[tweet_dst]
                    edges.append((src_idx, dst_idx))
    
    # 构建edge_index
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # 如果没有边，创建空的edge_index
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    # 创建初始节点特征（64维，随机初始化）
    # 注意：在实际应用中，这里应该使用LSTM编码文本，但第一阶段可以使用随机特征
    x = torch.randn(num_nodes, 64)
    
    # 创建Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
    )
    
    # 可选：存储额外的元数据（不用于训练，但可以用于调试）
    data.news_id = news_id
    data.title = title
    data.text = text
    
    return data


class MCFakeDataset(Dataset):
    """MC-Fake数据集类。"""
    
    def __init__(self, data_list):
        super().__init__()
        self._data_list = data_list
    
    def len(self):
        return len(self._data_list)
    
    def get(self, idx):
        return self._data_list[idx]


def load_mcfake(
    csv_path: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    random_seed: int = 42,
) -> tuple[MCFakeDataset, MCFakeDataset, MCFakeDataset]:
    """
    加载MC-Fake数据集并划分为训练/验证/测试集。
    
    Args:
        csv_path: CSV文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例（剩余部分为测试集）
        random_seed: 随机种子
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {csv_path}")
    
    # 读取CSV文件
    data_list = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        # 尝试检测CSV格式
        sample = f.read(1024)
        f.seek(0)
        
        # 检查是否有BOM
        if sample.startswith("\ufeff"):
            f.seek(3)
        
        reader = csv.DictReader(f)
        
        # 处理列名：可能拼写错误 "tittle" -> "title"
        fieldnames = reader.fieldnames
        if fieldnames:
            fieldname_map = {}
            for fn in fieldnames:
                if fn == "tittle":
                    fieldname_map["tittle"] = "title"
                else:
                    fieldname_map[fn] = fn
            
            for row in reader:
                # 处理列名映射
                normalized_row = {}
                for key, value in row.items():
                    if key == "tittle":
                        normalized_row["title"] = value
                    else:
                        normalized_row[key] = value
                
                # 确保所有必需的字段都存在
                required_fields = [
                    "news_id", "title", "url", "publish_date", "source", "text",
                    "labels", "n_tweets", "n_retweets", "n_replies", "n_users",
                    "tweet_ids", "retweet_ids", "reply_ids", "user_ids",
                    "retweet_relations", "reply_relations", "data_name"
                ]
                
                # 填充缺失字段
                for field in required_fields:
                    if field not in normalized_row:
                        normalized_row[field] = ""
                
                # 构建图
                data = build_graph_from_event(
                    news_id=normalized_row.get("news_id", ""),
                    title=normalized_row.get("title", normalized_row.get("tittle", "")),
                    url=normalized_row.get("url", ""),
                    publish_date=normalized_row.get("publish_date", ""),
                    source=normalized_row.get("source", ""),
                    text=normalized_row.get("text", ""),
                    labels=normalized_row.get("labels", "0"),
                    n_tweets=normalized_row.get("n_tweets", "0"),
                    n_retweets=normalized_row.get("n_retweets", "0"),
                    n_replies=normalized_row.get("n_replies", "0"),
                    n_users=normalized_row.get("n_users", "0"),
                    tweet_ids=normalized_row.get("tweet_ids", ""),
                    retweet_ids=normalized_row.get("retweet_ids", ""),
                    reply_ids=normalized_row.get("reply_ids", ""),
                    user_ids=normalized_row.get("user_ids", ""),
                    retweet_relations=normalized_row.get("retweet_relations", ""),
                    reply_relations=normalized_row.get("reply_relations", ""),
                    data_name=normalized_row.get("data_name", ""),
                )
                
                if data is not None:
                    data_list.append(data)
    
    if len(data_list) == 0:
        raise ValueError(f"未能从CSV文件加载任何有效数据: {csv_path}")
    
    # 随机打乱并划分数据集
    import random
    random.seed(random_seed)
    random.shuffle(data_list)
    
    total = len(data_list)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    
    train_data = data_list[:train_end]
    val_data = data_list[train_end:val_end]
    test_data = data_list[val_end:]
    
    train_dataset = MCFakeDataset(train_data)
    val_dataset = MCFakeDataset(val_data)
    test_dataset = MCFakeDataset(test_data)
    
    return train_dataset, val_dataset, test_dataset


__all__ = ["build_graph_from_event", "load_mcfake", "MCFakeDataset"]
