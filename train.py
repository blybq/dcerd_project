"""
阶段二训练入口：使用真实MC-Fake数据集训练模型。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.data.loader import load_mcfake
from src.models.dcerd import DCERD


def batch_to_data_list(batch):
    """
    将批处理的Batch对象分离为单独的Data对象列表。
    
    这是to_data_list的替代实现，兼容不同版本的PyTorch Geometric。
    """
    # 方法1: 尝试使用PyTorch Geometric的内置方法（如果存在）
    try:
        from torch_geometric.utils import to_data_list
        return to_data_list(batch)
    except ImportError:
        pass
    
    # 方法2: 尝试使用Batch对象的to_data_list方法（某些版本）
    if hasattr(batch, 'to_data_list'):
        return batch.to_data_list()
    
    # 方法3: 手动分离（兼容性最好的方法）
    data_list = []
    
    # 检查是否是批处理（有batch属性）
    if hasattr(batch, 'batch') and batch.batch is not None:
        # 批处理模式：使用batch属性分离
        num_graphs = batch.batch.max().item() + 1
        
        # 获取每个图的节点范围
        node_counts = torch.bincount(batch.batch, minlength=num_graphs)
        node_offsets = torch.cumsum(torch.cat([torch.tensor([0], device=node_counts.device), node_counts[:-1]]), dim=0)
        
        for i in range(num_graphs):
            start_idx = node_offsets[i].item()
            end_idx = start_idx + node_counts[i].item()
            
            # 提取节点特征
            x = batch.x[start_idx:end_idx] if hasattr(batch, 'x') and batch.x is not None else None
            
            # 提取边索引（PyTorch Geometric在批处理时已经调整了边索引）
            if hasattr(batch, 'edge_index') and batch.edge_index is not None and batch.edge_index.numel() > 0:
                # 找到属于当前图的边
                edge_mask = (batch.edge_index[0] >= start_idx) & (batch.edge_index[0] < end_idx) & \
                           (batch.edge_index[1] >= start_idx) & (batch.edge_index[1] < end_idx)
                edge_index = batch.edge_index[:, edge_mask]
                
                # 重新映射节点索引到局部索引（从0开始）
                if edge_index.numel() > 0:
                    edge_index = edge_index - start_idx
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long, device=batch.x.device if hasattr(batch, 'x') else torch.device('cpu'))
            
            # 提取标签
            if hasattr(batch, 'y') and batch.y is not None:
                if batch.y.dim() == 0:
                    # 单个标签（所有图共享）
                    y = batch.y
                elif batch.y.dim() == 1:
                    # 每个图一个标签
                    y = batch.y[i] if i < batch.y.shape[0] else None
                else:
                    y = None
            else:
                y = None
            
            # 创建Data对象
            if x is not None:
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
    else:
        # 单个图：直接返回
        data_list.append(batch)
    
    return data_list


def parse_args():
    parser = argparse.ArgumentParser(description="训练DCE-RD谣言检测模型")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mcfake",
        choices=["mcfake", "weibo"],
        help="数据集名称",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="训练轮数（覆盖配置文件）"
    )
    parser.add_argument(
        "--config", type=str, default="configs/mcfake.yaml", help="配置文件路径"
    )
    parser.add_argument(
        "--data-path", type=str, default=None, help="数据文件路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu，默认自动选择）",
    )
    parser.add_argument(
        "--save-dir", type=str, default="checkpoints", help="模型保存目录"
    )
    return parser.parse_args()


def load_config(path: str):
    """加载YAML配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_str: str | None) -> torch.device:
    """获取训练设备。"""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 设备设置
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 数据路径
    data_path = args.data_path or cfg.get("data", {}).get("path", "MC_Fake_dataset.csv")
    if not os.path.exists(data_path):
        # 尝试在项目根目录查找
        project_root = Path(__file__).parent
        alt_path = project_root / data_path
        if alt_path.exists():
            data_path = str(alt_path)
        else:
            # 尝试直接使用文件名
            alt_path = project_root / "MC_Fake_dataset.csv"
            if alt_path.exists():
                data_path = str(alt_path)
            else:
                raise FileNotFoundError(
                    f"找不到数据文件: {data_path}。请使用 --data-path 指定正确路径。"
                )

    print(f"加载数据集: {data_path}")
    train_dataset, val_dataset, test_dataset = load_mcfake(
        csv_path=data_path,
        train_ratio=0.8,
        val_ratio=0.1,
        random_seed=42,
    )

    print(
        f"数据集大小: 训练={len(train_dataset)}, "
        f"验证={len(val_dataset)}, 测试={len(test_dataset)}"
    )

    # 创建DataLoader
    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型初始化
    model = DCERD(cfg.get("model", {}))
    model = model.to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["lr"]
    )

    # 训练轮数
    num_epochs = args.epochs or cfg["training"]["epochs"]

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 训练循环
    print("\n开始训练...")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)
            
            # 将批处理分离为单独的图
            data_list = batch_to_data_list(batch)
            
            for data in data_list:
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                label = data.y.to(device)

                # 前向传播
                outputs = model(x, edge_index)
                logits = outputs["logits"]
                loss = model.compute_loss(outputs, label)

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 统计
                train_loss += loss.item()
                pred = logits.argmax(dim=-1)
                train_correct += (pred == label).sum().item()
                train_total += 1

        avg_train_loss = train_loss / max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                data_list = batch_to_data_list(batch)
                
                for data in data_list:
                    x = data.x.to(device)
                    edge_index = data.edge_index.to(device)
                    label = data.y.to(device)

                    outputs = model(x, edge_index)
                    logits = outputs["logits"]
                    loss = model.compute_loss(outputs, label)

                    val_loss += loss.item()
                    pred = logits.argmax(dim=-1)
                    val_correct += (pred == label).sum().item()
                    val_total += 1

        avg_val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": cfg,
            }
            torch.save(checkpoint, save_dir / "best_model.pt")
            print(f"  -> 保存最佳模型 (Val Acc: {val_acc:.4f})")

    print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

