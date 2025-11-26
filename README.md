# DCE-RD 谣言检测复现

本仓库实现 KDD 2023 论文《Rumor Detection with Diverse Counterfactual Evidence》的完整复现。

## 项目概述

DCE-RD (Diverse Counterfactual Evidence for Rumor Detection) 是一个基于图神经网络的谣言检测模型，使用多样化的反事实证据来提高检测性能。

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric 2.0+

### 安装依赖

```bash
pip install -r requirements.txt
```

### 第一阶段验证（合成数据）

```bash
cd dcerd_project
python main.py  # 应该输出: Phase 1 integration test passed!
```

### 最小化训练/测试验证

```bash
python -m pytest tests/test_minimal_training.py -v
```

## 数据集准备

**重要**: 由于数据集文件较大（MC_Fake_dataset.csv），**不应提交到GitHub仓库**。

### 本地使用

1. 将 `MC_Fake_dataset.csv` 文件放置在 `dcerd_project/` 目录下
2. 确保文件格式符合要求（18列，包含 tweet_ids, retweet_relations, reply_relations, labels 等字段）

### Google Colab 使用

1. 将代码推送到GitHub（不包含CSV文件）
2. 在Colab中打开 `colab_setup.ipynb`
3. 按照Notebook中的步骤上传数据集文件

## 训练模型

### 本地训练

```bash
# 使用默认配置
python train.py

# 指定参数
python train.py --epochs 20 --data-path MC_Fake_dataset.csv --device cuda

# 使用自定义配置
python train.py --config configs/mcfake.yaml --epochs 10
```

训练过程中会：
- 自动划分数据集（8:1:1）
- 保存最佳模型到 `checkpoints/best_model.pt`
- 输出每个epoch的训练/验证损失和准确率

### Google Colab 训练

1. 打开 `colab_setup.ipynb`
2. 按照Notebook中的步骤执行
3. 上传数据集文件
4. 运行训练单元格

## 评估模型

```bash
# 评估测试集性能
python evaluate.py --weights checkpoints/best_model.pt --data-path MC_Fake_dataset.csv

# 指定设备
python evaluate.py --weights checkpoints/best_model.pt --device cuda
```

评估脚本会输出：
- 准确率 (Accuracy)
- AUC-ROC
- 混淆矩阵
- 精确率、召回率、F1分数

## 运行测试

```bash
# 运行所有测试
python -m pytest tests/ --cov=src --cov-report=term

# 运行最小化训练测试
python -m pytest tests/test_minimal_training.py -v

# 代码格式检查
black --check src tests

# 类型检查
mypy src
```

## 项目结构

```
dcerd_project/
├── src/                    # 源代码
│   ├── data/              # 数据加载
│   ├── models/            # 模型定义
│   ├── losses/            # 损失函数
│   └── utils/             # 工具函数
├── tests/                 # 测试代码
├── configs/               # 配置文件
├── checkpoints/           # 模型权重（不提交到Git）
├── main.py               # 第一阶段验证脚本
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── colab_setup.ipynb     # Colab训练脚本（不提交到Git）
├── requirements.txt      # 依赖列表
└── README.md             # 本文件
```

## 配置说明

配置文件位于 `configs/` 目录：

- `mcfake.yaml`: MC-Fake数据集配置
- `weibo.yaml`: Weibo数据集配置（可选）

主要参数：
- `model.hidden_dim`: 隐藏层维度（默认64）
- `model.K`: Top-K采样大小（默认15）
- `model.m`: 子图采样次数（默认3）
- `training.batch_size`: 批处理大小（默认4）
- `training.lr`: 学习率（默认0.001）

## 性能指标

论文报告的性能（参考）：
- MC-Fake: Accuracy ≥ 0.90, AUC ≥ 0.96
- Weibo: Accuracy ≥ 0.93, AUC ≥ 0.98

## 常见问题

### 1. 数据集文件太大，无法提交到GitHub？

**解决方案**: 
- 将 `MC_Fake_dataset.csv` 添加到 `.gitignore`（已包含）
- 在Colab中通过文件上传功能上传数据集
- 或使用Git LFS（如果必须版本控制）

### 2. 如何在Colab中使用？

1. 将代码推送到GitHub（不包含CSV和模型文件）
2. 在Colab中打开 `colab_setup.ipynb`
3. 修改GitHub仓库URL
4. 按照Notebook步骤执行

### 3. 训练时内存不足？

- 减小 `batch_size`
- 减小 `model.K` 和 `model.m`
- 使用更小的 `hidden_dim`

## 开发文档

详见 `开发文档/` 目录中的详细需求规格说明：
- `DCE-RD模型的分层实现.md`: 模型架构详解
- `DCE-RD谣言检测系统 - 分阶段开发需求规格说明书.md`: 完整开发规范

## 许可证

本项目仅用于学术研究目的。

