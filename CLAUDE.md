# 3D CT 影像分类项目 (VSNet)

## 项目概述

基于 **VSNet**（Swin Transformer + CNN 混合架构）的 3D CT 医学影像二分类项目，用于区分 **FD** 与 **OF** 两类 CT 影像。

## 文件结构

| 文件 | 职责 |
|------|------|
| `VSNet.py` | VSNet 模型定义，包含 Swin Transformer 编码器、注意力机制(CSA/SSA)、解码器和多任务输出头 |
| `train.py` | 训练主流程：数据加载、训练/测试集拆分(8:2)、VSNetClassifier 分类封装、训练循环与评估 |
| `preprocess.py` | DICOM → NIfTI 格式转换预处理脚本 |
| `best_model.pth` | 训练过程中保存的最优模型权重 |

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

train.py 中的 VSNetClassifier:
  VSNet 编码器 + Swin → 全局平均池化 → FC(192→2) → 二分类
```

## 运行方式

```bash
# 1. 预处理：将 DICOM 转为 NIfTI
python preprocess.py

# 2. 训练分类模型
python train.py
```

训练输出包括：每个 epoch 的训练损失、测试准确率（整体 + FD/OF 各类别），以及最终测试集评估报告。
