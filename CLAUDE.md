# 3D CT 影像分类项目 (VSNet)

## 项目概述

基于 **VSNet**（Swin Transformer + CNN 混合架构）的 3D CT 医学影像二分类项目，用于区分 **FD** 与 **OF** 两类 CT 影像。

## 文件结构

| 文件 | 职责 |
|------|------|
| `VSNet.py` | VSNet 模型定义，包含 Swin Transformer 编码器、注意力机制(CSA/SSA)、解码器和多任务输出头 |
| `preprocess.py` | 第一步：DICOM → NIfTI 格式转换 |
| `augment.py` | 第二步：离线数据增强（翻转/旋转/剪切/噪声/对比度/平滑），默认 10 倍扩充，支持断点续跑 |
| `transfer_learning.py` | 第三步（主训练脚本）：迁移学习训练，8:1:1 三路划分，支持 MONAI 预训练权重 + 渐进式解冻 |
| `classify.py` | 推理脚本：支持目录/单文件/测试集三种模式，自动检测标签并计算准确率 |
| `best_model.pth` | 训练过程中保存的最优模型权重（已弃用，请用 best_model_tl.pth） |
| `best_model_tl.pth` | 迁移学习保存的最优验证集模型权重 |
| `test_split.json` | 训练时自动生成的测试集文件列表（classify.py --test_split 读取） |

## 关键依赖

- **PyTorch** 2.0+（使用 `torch.amp` 混合精度训练）
- **MONAI**（医学影像数据加载、变换、网络组件）
- **SimpleITK** / **pydicom**（DICOM 读取与转换）
- **einops**（张量维度变换）

## 架构概要

```
VSNet (backbone)
  ├── ResU-Net 编码器 (4层)
  ├── Swin Transformer 瓶颈层
  ├── CSA + SSA 注意力模块
  └── ResU-Net 解码器 + 多任务输出

VSNetClassifier (分类封装):
  VSNet 编码器 + Swin → 全局平均池化 → FC(192→2) → 二分类
```

## 运行方式

```bash
# 第一步：DICOM → NIfTI（一次性）
python preprocess.py

# 第二步：数据增强（默认 10 倍，支持断点续跑，一次性）
python augment.py

# 第三步：训练（自动 8:1:1 划分训练/验证/测试集）
python transfer_learning.py                     # 迁移学习（推荐）
python transfer_learning.py --no_pretrain       # 从头训练（对比实验）

# 推理：对测试集评估（训练完全不可见）
python classify.py --model best_model_tl.pth --test_split test_split.json

# 推理：对新数据目录分类
python classify.py --model best_model_tl.pth --data_dir <新数据目录>
python classify.py --model best_model_tl.pth --single <单个.nii.gz文件>
```

训练输出包括：每个 epoch 的训练损失、验证集准确率（整体 + FD/OF 各类别）。测试集不参与训练，需用 `classify.py --test_split` 独立评估。

## 数据集划分

训练时自动按 **8:1:1** 分层划分（按 FD/OF 类别分别 shuffle 后切分）：

| 子集 | 比例 | 用途 |
|------|------|------|
| 训练集 | 80% | 喂给模型训练 |
| 验证集 | 10% | 每 epoch 后评估，挑选最优模型 |
| 测试集 | 10% | 训练完全不可见，写入 `test_split.json` 供 `classify.py` 独立评估 |
