# 3D CT 影像分类 (VSNet)

基于 **VSNet**（Swin Transformer + CNN 混合架构）的 3D CT 医学影像二分类项目，用于区分 **FD** 与 **OF** 两类 CT 影像。

## 模型架构

```
VSNet (backbone)
  ├── ResU-Net CNN 编码器 (4 层，feature_size=12)
  ├── Swin Transformer 瓶颈层 (depth=2, window_size=7)
  ├── CSA (Channel Self-Attention) + SSA (Spatial Self-Attention)
  └── Global Avg Pool → FC(192→64→2) → 二分类
```

- **模式**: `mode="classification"`（纯分类，不含 Decoder）
- **输入**: 96×96×96 单通道 CT 体素
- **预训练**: MONAI SwinUNETR (BTCV) 权重初始化 Swin Transformer 层
- **训练策略**: 渐变解冻 + 差分学习率 (CNN 1× / Swin 10× / Head 100×)

## 文件结构

| 文件 | 职责 |
|------|------|
| `VSNet.py` | VSNet 模型定义（含 classification/segmentation 双模式） |
| `preprocess.py` | 第一步：DICOM → NIfTI 格式转换 |
| `augment.py` | 第二步：离线数据增强（翻/转/剪切/噪声/对比度），10 倍扩充 |
| `transfer_learning.py` | 第三步：迁移学习训练，8:1:1 划分，支持 K-Fold |
| `classify.py` | 第四步：推理脚本，三种模式（目录/单文件/测试集） |
| `plot_slices.py` | 可视化：数据增强前后 3D 切片对比 |
| `plot_training.py` | 可视化：迁移学习 vs 从头训练学习曲线 |
| `plot_metrics.py` | 可视化：混淆矩阵热力图 + ROC-AUC 曲线 |
| `plots/` | 所有可视化输出目录 |

## 环境依赖

```bash
conda create -n med3d python=3.10
conda activate med3d
pip install torch monai SimpleITK pydicom nibabel einops matplotlib seaborn scikit-learn tqdm
```

核心依赖：
- **PyTorch** 2.0+（`torch.amp` 混合精度）
- **MONAI** 1.4+（医学影像数据加载、网络组件、预训练权重）
- **SimpleITK** / **pydicom**（DICOM/NIfTI 读写）
- **einops**（张量维度变换）
- **matplotlib / seaborn**（可视化）

## 使用流程

### 第一步：DICOM → NIfTI（一次性）

```bash
python preprocess.py
# 输入: DICOM 目录 → 输出: dataset/NIfTI_Data/{FD,OF}/*.nii.gz
```

### 第二步：数据增强（一次性）

```bash
python augment.py
# 每原始样本生成 10 个增强版本 (aug0~aug9)
# 输出 dataset/NIfTI_Data/augment_manifest.json
```

### 第三步：训练

```bash
# 迁移学习（推荐）—— Swin 层加载 MONAI 预训练权重
python transfer_learning.py

# 从头训练（对比实验）
python transfer_learning.py --no_pretrain --save_path best_model_scratch.pth --log_file training_log_scratch.json

# 调整超参
python transfer_learning.py --epochs 60 --lr 1e-4 --backbone_lr_factor 0.1 --batch_size 4
```

训练过程会自动：
- 8:1:1 分层划分（按原始样本分组，防止增强数据泄漏）
- 生成 `test_split.json`（测试集文件列表）
- 每 epoch 保存验证集最优模型
- 保存训练日志 JSON（供可视化）

### 第四步：推理

```bash
# 对测试集评估（训练完全不可见）
python classify.py --model best_model_tl.pth --test_split test_split.json

# 对新数据目录分类（有 FD/OF 子目录则自动计算准确率）
python classify.py --model best_model_tl.pth --data_dir F:/path/to/new_data

# 单文件推理
python classify.py --model best_model_tl.pth --single F:/path/to/sample.nii.gz

# 保存结果为 JSON（供可视化）
python classify.py --model best_model_tl.pth --test_split test_split.json --save_json results.json
```

### 第五步：可视化

```bash
# 数据增强前后对比（3 个正交切面）
python plot_slices.py --source FD_1 --aug_idx 0

# 学习曲线对比（迁移学习 vs 从头训练）
python plot_training.py --log_tl training_log.json --log_scratch training_log_scratch.json

# 混淆矩阵 + ROC-AUC
python plot_metrics.py --results results.json
```

所有图片输出到 `plots/` 目录。

## 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lr` | 2e-4 | 分类头初始学习率 |
| `--epochs` | 40 | 训练总轮数 |
| `--batch_size` | 4 | 批次大小 |
| `--dropout` | 0.3 | CSA/SSA/FC dropout |
| `--weight_decay` | 1e-2 | AdamW 权重衰减 |
| `--backbone_lr_factor` | 0.01 | CNN encoder 学习率 = base_lr × 此值 |
| `--swin_lr_factor` | 0.1 | Swin+Attention 学习率 = base_lr × 此值 |
| `--kfold` | 5 | K-Fold 折数（0 或 1 为单次划分） |
| `--no_pretrain` | False | 不使用预训练权重，随机初始化 |

## 数据集

- 类别: FD / OF（二分类）
- 原始样本: 27 例（FD 15 / OF 12）
- 增强后: ~297 个文件（含原始 + 10×增强）
- 输入尺寸: 96×96×96（自动 Resize）
- 划分: 80% 训练 / 10% 验证 / 10% 测试（按原始样本分组防泄漏）

## 已知注意事项

- `classify.py` 使用 SimpleITK 直接加载 `.nii.gz`，不依赖 MONAI reader
- `transfer_learning.py` 使用 `NiBabelReader` 加载 NIfTI 文件
- 预训练权重首次运行时会自动从 MONAI Model Zoo 下载到 `pretrained_models/`
- `freeze_cnn=False`：CNN Encoder 从 epoch 1 开始训练（之前有个 bug 是冻结导致 50% 准确率）
