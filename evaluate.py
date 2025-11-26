"""
测试集评估脚本：在真实测试集上评估模型性能。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_data_list

from src.data.loader import load_mcfake
from src.models.dcerd import DCERD
from src.utils.metrics import accuracy, binary_auc


def parse_args():
    parser = argparse.ArgumentParser(description="评估DCE-RD模型在测试集上的性能")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="模型权重文件路径（checkpoints/best_model.pt）",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="测试数据CSV文件路径（默认使用训练时的数据路径）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mcfake.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu，默认自动选择）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="批处理大小",
    )
    return parser.parse_args()


def get_device(device_str: str | None) -> torch.device:
    """获取评估设备。"""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(path: str):
    """加载YAML配置文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    device = get_device(args.device)
    print(f"使用设备: {device}")

    # 加载配置
    cfg = load_config(args.config)

    # 加载模型
    print(f"加载模型权重: {args.weights}")
    checkpoint = torch.load(args.weights, map_location=device)
    model_config = checkpoint.get("config", {}).get("model", cfg.get("model", {}))
    model = DCERD(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"模型已加载（训练轮数: {checkpoint.get('epoch', 'unknown')}）")

    # 加载测试数据
    data_path = args.data_path or cfg.get("data", {}).get("path", "MC_Fake_dataset.csv")
    if not os.path.exists(data_path):
        project_root = Path(__file__).parent
        alt_path = project_root / data_path
        if alt_path.exists():
            data_path = str(alt_path)
        else:
            alt_path = project_root / "MC_Fake_dataset.csv"
            if alt_path.exists():
                data_path = str(alt_path)
            else:
                raise FileNotFoundError(
                    f"找不到数据文件: {data_path}。请使用 --data-path 指定正确路径。"
                )

    print(f"加载测试数据集: {data_path}")
    _, _, test_dataset = load_mcfake(
        csv_path=data_path,
        train_ratio=0.8,
        val_ratio=0.1,
        random_seed=42,
    )
    print(f"测试集大小: {len(test_dataset)} 个样本")

    # 创建DataLoader
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # 评估
    print("\n开始评估...")
    all_preds = []
    all_labels = []
    all_probs = []
    test_loss = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            data_list = to_data_list(batch)

            for data in data_list:
                x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                label = data.y.to(device)

                outputs = model(x, edge_index)
                logits = outputs["logits"]
                loss = model.compute_loss(outputs, label)

                # 计算概率
                probs = torch.softmax(logits, dim=-1)[:, 1]  # 取正类概率

                # 统计
                test_loss += loss.item()
                pred = logits.argmax(dim=-1)
                test_correct += (pred == label).sum().item()
                test_total += 1

                # 收集所有预测和标签
                all_preds.append(pred.cpu())
                all_labels.append(label.cpu())
                all_probs.append(probs.cpu())

    # 汇总结果
    all_preds_tensor = torch.cat(all_preds)
    all_labels_tensor = torch.cat(all_labels)
    all_probs_tensor = torch.cat(all_probs)

    avg_loss = test_loss / max(test_total, 1)
    test_acc = test_correct / max(test_total, 1)
    test_auc = binary_auc(all_probs_tensor, all_labels_tensor)

    # 打印结果
    print("\n" + "=" * 60)
    print("测试集评估结果")
    print("=" * 60)
    print(f"测试样本数: {test_total}")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率 (Accuracy): {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"AUC-ROC: {test_auc:.4f}")
    print("=" * 60)

    # 计算混淆矩阵
    true_pos = ((all_preds_tensor == 1) & (all_labels_tensor == 1)).sum().item()
    true_neg = ((all_preds_tensor == 0) & (all_labels_tensor == 0)).sum().item()
    false_pos = ((all_preds_tensor == 1) & (all_labels_tensor == 0)).sum().item()
    false_neg = ((all_preds_tensor == 0) & (all_labels_tensor == 1)).sum().item()

    print("\n混淆矩阵:")
    print(f"  真正例 (TP): {true_pos}")
    print(f"  真负例 (TN): {true_neg}")
    print(f"  假正例 (FP): {false_pos}")
    print(f"  假负例 (FN): {false_neg}")

    if true_pos + false_neg > 0:
        recall = true_pos / (true_pos + false_neg)
        print(f"  召回率 (Recall): {recall:.4f}")
    if true_pos + false_pos > 0:
        precision = true_pos / (true_pos + false_pos)
        print(f"  精确率 (Precision): {precision:.4f}")
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            print(f"  F1分数: {f1:.4f}")


if __name__ == "__main__":
    main()

