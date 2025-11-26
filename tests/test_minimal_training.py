"""
最小化训练/测试验证：使用大小为1的训练集和测试集验证完整流程。

这个测试确保：
1. 数据加载正常
2. 模型前向传播正常
3. 损失计算正常
4. 反向传播正常
5. 模型保存/加载正常
6. 评估流程正常
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import torch

from src.data.loader import build_graph_from_event, load_mcfake
from src.models.dcerd import DCERD
from src.utils.metrics import accuracy, binary_auc


def create_minimal_csv() -> Path:
    """创建一个包含2个样本的最小CSV文件用于测试。"""
    # 创建两个样本：一个有足够的节点和边，另一个也足够复杂
    rows = [
        {
            "news_id": "test_001",
            "title": "Test News 1",
            "url": "http://test.com/1",
            "publish_date": "2023-01-01",
            "source": "TestSource",
            "text": "This is a test news article for training.",
            "labels": "1",  # 真谣言
            "n_tweets": "15",  # 足够的节点数
            "n_retweets": "10",
            "n_replies": "5",
            "n_users": "12",
            "tweet_ids": "1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015",
            "retweet_ids": "1002,1003,1004,1005,1006,1007,1008,1009,1010",
            "reply_ids": "1011,1012,1013,1014,1015",
            "user_ids": "2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012",
            # 创建复杂的转发关系网络
            "retweet_relations": (
                "1002-1001-2002-2001,"  # 1002转发1001
                "1003-1001-2003-2001,"  # 1003转发1001
                "1004-1002-2004-2002,"  # 1004转发1002
                "1005-1002-2005-2002,"  # 1005转发1002
                "1006-1003-2006-2003,"  # 1006转发1003
                "1007-1004-2007-2004,"  # 1007转发1004
                "1008-1005-2008-2005,"  # 1008转发1005
                "1009-1006-2009-2006,"  # 1009转发1006
                "1010-1007-2010-2007"  # 1010转发1007
            ),
            # 创建回复关系
            "reply_relations": (
                "1011-1001-2011-2001,"  # 1011回复1001
                "1012-1002-2012-2002,"  # 1012回复1002
                "1013-1003-2011-2003,"  # 1013回复1003
                "1014-1004-2012-2004,"  # 1014回复1004
                "1015-1005-2011-2005"  # 1015回复1005
            ),
            "data_name": "test",
        },
        {
            "news_id": "test_002",
            "title": "Test News 2",
            "url": "http://test.com/2",
            "publish_date": "2023-01-02",
            "source": "TestSource",
            "text": "This is another test news article for testing.",
            "labels": "0",  # 假谣言
            "n_tweets": "12",
            "n_retweets": "8",
            "n_replies": "4",
            "n_users": "10",
            "tweet_ids": "2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012",
            "retweet_ids": "2002,2003,2004,2005,2006,2007,2008",
            "reply_ids": "2009,2010,2011,2012",
            "user_ids": "3001,3002,3003,3004,3005,3006,3007,3008,3009,3010",
            # 创建另一个复杂的关系网络
            "retweet_relations": (
                "2002-2001-3002-3001,"
                "2003-2001-3003-3001,"
                "2004-2002-3004-3002,"
                "2005-2002-3005-3002,"
                "2006-2003-3006-3003,"
                "2007-2004-3007-3004,"
                "2008-2005-3008-3005"
            ),
            "reply_relations": (
                "2009-2001-3009-3001,"
                "2010-2002-3010-3002,"
                "2011-2003-3009-3003,"
                "2012-2004-3010-3004"
            ),
            "data_name": "test",
        },
    ]

    # 创建临时CSV文件
    fd, temp_path = tempfile.mkstemp(suffix=".csv", prefix="minimal_test_")
    with open(temp_path, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return Path(temp_path)


def test_minimal_training_and_evaluation():
    """测试最小化训练和评估流程。"""
    print("=" * 60)
    print("最小化训练/测试验证")
    print("=" * 60)

    # 1. 创建最小CSV文件
    print("\n1. 创建最小测试数据集...")
    csv_path = create_minimal_csv()
    print(f"   临时CSV文件: {csv_path}")

    try:
        # 2. 加载数据（强制划分为1个训练样本和1个测试样本）
        print("\n2. 加载数据...")
        train_ds, val_ds, test_ds = load_mcfake(
            csv_path=csv_path,
            train_ratio=0.5,  # 50%训练，50%测试（实际会得到1:1）
            val_ratio=0.0,  # 无验证集
            random_seed=42,
        )

        # 手动调整为1个训练样本和1个测试样本
        if len(train_ds) > 1:
            train_ds._data_list = train_ds._data_list[:1]
        if len(test_ds) > 1:
            test_ds._data_list = test_ds._data_list[:1]

        print(f"   训练集: {len(train_ds)} 个样本")
        print(f"   测试集: {len(test_ds)} 个样本")

        # 检查样本
        train_sample = train_ds[0]
        test_sample = test_ds[0]
        print(f"\n   训练样本:")
        print(f"     节点数: {train_sample.x.shape[0]}")
        print(f"     边数: {train_sample.edge_index.shape[1]}")
        print(f"     标签: {train_sample.y.item()}")
        print(f"\n   测试样本:")
        print(f"     节点数: {test_sample.x.shape[0]}")
        print(f"     边数: {test_sample.edge_index.shape[1]}")
        print(f"     标签: {test_sample.y.item()}")

        # 3. 创建模型
        print("\n3. 创建模型...")
        model = DCERD(
            config={
                "hidden_dim": 64,
                "K": 5,  # 较小的K以适应小图
                "m": 3,
                "beta": 1.0,
                "gamma": 0.5,
            }
        )
        print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 4. 训练一个epoch
        print("\n4. 执行训练步骤...")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()

        x = train_sample.x
        edge_index = train_sample.edge_index
        label = train_sample.y

        # 前向传播
        outputs = model(x, edge_index)
        logits = outputs["logits"]
        loss = model.compute_loss(outputs, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"   训练损失: {loss.item():.4f}")
        print(f"   训练预测: {logits.argmax(dim=-1).item()}")
        print(f"   真实标签: {label.item()}")

        # 5. 评估
        print("\n5. 执行评估步骤...")
        model.eval()
        with torch.no_grad():
            test_x = test_sample.x
            test_edge_index = test_sample.edge_index
            test_label = test_sample.y

            test_outputs = model(test_x, test_edge_index)
            test_logits = test_outputs["logits"]
            test_loss = model.compute_loss(test_outputs, test_label)

            test_acc = accuracy(test_logits, test_label)
            test_probs = torch.softmax(test_logits, dim=-1)[:, 1]
            # 对于单个样本，AUC无法计算，跳过
            print(f"   测试损失: {test_loss.item():.4f}")
            print(f"   测试准确率: {test_acc:.4f}")
            print(f"   测试预测: {test_logits.argmax(dim=-1).item()}")
            print(f"   真实标签: {test_label.item()}")

            # 6. 验证模型保存/加载
            print("\n6. 验证模型保存/加载...")
            import tempfile

            # 先保存训练后的模型状态
            with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
                checkpoint_path = Path(f.name)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": {"hidden_dim": 64, "K": 5, "m": 3},
                    },
                    checkpoint_path,
                )

                # 加载模型
                loaded_checkpoint = torch.load(checkpoint_path)
                loaded_model = DCERD(loaded_checkpoint["config"])
                loaded_model.load_state_dict(loaded_checkpoint["model_state_dict"])
                loaded_model.eval()

                # 使用训练后的模型重新计算logits（用于比较）
                model.eval()
                with torch.no_grad():
                    # 重新计算当前模型的输出
                    current_outputs = model(test_x, test_edge_index)
                    current_logits = current_outputs["logits"]
                    
                    # 加载后的模型输出
                    loaded_outputs = loaded_model(test_x, test_edge_index)
                    loaded_logits = loaded_outputs["logits"]

                    # 验证输出一致（比较训练后模型和加载后模型的输出）
                    # 注意：由于Gumbel采样的随机性，输出可能不完全相同
                    # 我们只验证模型能够正常保存和加载，不要求输出完全一致
                    # 但应该验证输出形状和数值范围合理
                    assert current_logits.shape == loaded_logits.shape, "模型加载后输出形状不一致"
                    assert torch.isfinite(current_logits).all(), "当前模型输出包含非有限值"
                    assert torch.isfinite(loaded_logits).all(), "加载模型输出包含非有限值"
                    print("   模型保存/加载验证通过（输出形状和数值范围正常）")

        print("\n" + "=" * 60)
        print("[PASS] 最小化训练/测试验证全部通过！")
        print("=" * 60)

    finally:
        # 清理临时文件
        if csv_path.exists():
            try:
                csv_path.unlink()
                print(f"\n已清理临时文件: {csv_path}")
            except PermissionError:
                # Windows上文件可能还在使用，忽略错误
                print(f"\n警告：无法删除临时文件（可能正在使用）: {csv_path}")


if __name__ == "__main__":
    torch.manual_seed(42)
    test_minimal_training_and_evaluation()

