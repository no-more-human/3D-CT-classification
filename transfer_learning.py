"""
迁移学习训练脚本 —— 使用 MONAI 预训练模型初始化 VSNet，解决小样本过拟合问题。

策略：
  1. 优先尝试 MONAI Model Zoo 的 SwinUNETR 预训练权重
  2. 若下载失败，退化为 Kaiming 初始化 + 强力正则化训练
  3. 渐进式解冻：先训练分类头 → 再微调顶层 → 最后全模型微调
  4. 差分学习率：底层小 / 顶层大
  5. 读取 augment_manifest.json 按原始样本分组划分，防止数据泄漏
  6. AdamW 自动对 bias / LayerNorm / InstanceNorm 排除 weight decay
  7. CrossEntropyLoss 按类别样本数自动计算 class_weight

用法：
  python transfer_learning.py                    # 默认参数
  python transfer_learning.py --lr 1e-4 --epochs 30  # 自定义参数
"""
import os
import glob
import json
import random

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, RandRotate90d, RandFlipd
)
from monai.data import Dataset

from VSNet import VSNet

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="VSNet 迁移学习训练")
parser.add_argument("--lr", type=float, default=2e-4, help="初始学习率")
parser.add_argument("--epochs", type=int, default=40, help="训练总轮数")
parser.add_argument("--freeze_epochs", type=int, default=5,
                    help="冻结 backbone 的轮数（先训练分类头）")
parser.add_argument("--batch_size", type=int, default=1, help="批次大小")
parser.add_argument("--data_dir", type=str,
                    default=r"F:\python\3DCT_Classification\dataset\NIfTI_Data",
                    help="数据集路径")
parser.add_argument("--image_size", type=int, default=96, help="输入尺寸")
parser.add_argument("--no_pretrain", action="store_true",
                    help="不使用预训练权重（对比实验用）")
parser.add_argument("--save_path", type=str, default="best_model_tl.pth",
                    help="模型保存路径")
args = parser.parse_args()


# ==========================================
# 1. 分类器封装（同 train.py）
# ==========================================
class VSNetClassifier(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super().__init__()
        self.backbone = original_model
        in_features = 16 * 12  # feature_size=12 → 192
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x1 = self.backbone.resuetencoder1(x)
        x2 = self.backbone.resuetencoder2(x1)
        x2 = self.backbone.pool2(x2)
        x1 = self.backbone.gate2(x1, x2)
        x3 = self.backbone.resuetencoder3(x2)
        x3 = self.backbone.pool3(x3)
        x2 = self.backbone.gate3(x2, x3)
        x4 = self.backbone.resuetencoder4(x3)
        x4 = self.backbone.pool4(x4)
        x3 = self.backbone.gate4(x3, x4)

        x5 = self.backbone.swintransformer(x4.contiguous())
        x5 = self.backbone.CSA(x5)
        x5 = self.backbone.SSA(x5)

        out = self.global_pool(x5)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ==========================================
# 2. 预训练权重加载（MONAI Model Zoo）
# ==========================================
def load_pretrained_weights(model, device):
    """
    尝试从 MONAI Model Zoo 加载 SwinUNETR 预训练权重，映射到 VSNet 兼容层。
    若失败则退化为随机初始化 + 强力正则化。
    """
    pretrained_loaded = False

    # ----- MONAI Bundle -----
    try:
        from monai.bundle import load
        print("[迁移学习] 尝试从 MONAI Model Zoo 下载 SwinUNETR 预训练权重...")
        # SwinUNETR BTCV 多器官分割模型
        pretrained = load(
            name="swin_unetr_btcv_segmentation",
            bundle_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models"),
            source="github",
        )
        if pretrained is not None:
            state_dict = pretrained.state_dict() if hasattr(pretrained, "state_dict") else pretrained
            _map_swin_weights(model, state_dict)
            pretrained_loaded = True
            print("[迁移学习] [OK] 成功加载 MONAI SwinUNETR 预训练权重")
    except Exception as e:
        print(f"[迁移学习] MONAI Bundle 加载失败: {e}")

    # ----- 退化为 Kaiming 初始化 -----
    if not pretrained_loaded:
        print("[迁移学习] [WARN] 预训练模型加载失败，启用退化策略：Kaiming 初始化 + 强力正则化")
        _apply_kaiming_init(model)
        warnings.warn(
            "预训练权重加载失败，将在随机初始化下训练。"
            "建议运行 augment.py 扩充数据后再训练。"
        )

    return pretrained_loaded


def _map_swin_weights(model, pretrained_dict):
    """
    【修复版】将 SwinUNETR 的预训练权重映射到 VSNet 的 Swin Transformer 瓶颈层。
    """
    if hasattr(pretrained_dict, "state_dict"):
        pretrained_dict = pretrained_dict.state_dict()

    model_dict = model.state_dict()
    mapped = 0

    for k_model in model_dict:
        if "swintransformer" not in k_model:
            continue

        # 去掉模型网络层名中的 "backbone." 前缀，以便和预训练权重匹配
        k_pure = k_model.replace("backbone.", "")

        # 尝试匹配：将自己的 swintransformer 映射到官方的 swinViT.layers3
        candidate = k_pure.replace("swintransformer", "swinViT.layers3")
        if candidate in pretrained_dict and \
           model_dict[k_model].shape == pretrained_dict[candidate].shape:
            model_dict[k_model] = pretrained_dict[candidate]
            mapped += 1
            continue

        # 尝试去掉前缀后直接同名匹配
        if k_pure in pretrained_dict and \
           model_dict[k_pure].shape == pretrained_dict[k_pure].shape:
            model_dict[k_model] = pretrained_dict[k_pure]
            mapped += 1

    model.load_state_dict(model_dict, strict=False)
    print(f"[迁移学习]   已映射 {mapped} 个层的预训练权重 （共 {len(model_dict)} 层）")


def _apply_kaiming_init(model):
    """对所有 Conv3d 和 Linear 层应用 Kaiming 初始化（比默认初始化收敛更快）。"""
    for m in model.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


# ==========================================
# 3. 渐进式解冻工具
# ==========================================
def _split_params_by_weight_decay(named_params, lr, weight_decay):
    """
    将参数拆分为两组：需要 weight decay 的（weight 张量），
    和不需要的（bias、LayerNorm/InstanceNorm 的 weight）。
    经验上 weight decay 不应作用在 bias/norm 层，否则会降低模型性能。
    """
    decay_params = []
    nodecay_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or "norm" in name.lower() or name.endswith(".bias"):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params, "lr": lr, "weight_decay": weight_decay},
        {"params": nodecay_params, "lr": lr, "weight_decay": 0.0},
    ]


def freeze_backbone(model, freeze=True):
    """冻结 / 解冻 VSNet backbone（除 fc 分类头外）"""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = not freeze


def build_optimizer(model_or_param_groups, lr, weight_decay=1e-4):
    """创建 AdamW 优化器，自动对 bias/norm 排除 weight decay。"""
    if isinstance(model_or_param_groups, list):
        # 已分组（如差分学习率），对每组分别拆分 bias/norm
        all_groups = []
        for group in model_or_param_groups:
            named = [(f"group_{len(all_groups)}", p) for p in group["params"]]
            all_groups.extend(_split_params_by_weight_decay(
                named, group.get("lr", lr), group.get("weight_decay", weight_decay)))
        return torch.optim.AdamW(all_groups)
    else:
        params = _split_params_by_weight_decay(
            model_or_param_groups.named_parameters(), lr, weight_decay)
        return torch.optim.AdamW(params)


def setup_differential_lr(model, base_lr, backbone_factor=0.1):
    """
    差分学习率：backbone 用小学习率，分类头用基础学习率。
    避免破坏预训练权重的同时快速训练分类器。
    自动对 bias/norm 层排除 weight decay。
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "fc" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

    # 对 backbone 和 head 分别拆分 decay / no-decay
    all_groups = []
    all_groups.extend(_split_params_by_weight_decay(
        ((f"bb_{i}", p) for i, p in enumerate(backbone_params)),
        base_lr * backbone_factor, 1e-4))
    all_groups.extend(_split_params_by_weight_decay(
        ((f"hd_{i}", p) for i, p in enumerate(head_params)),
        base_lr, 1e-4))

    optimizer = torch.optim.AdamW(all_groups)

    print(f"[差分学习率] backbone: {base_lr * backbone_factor:.1e} | head: {base_lr:.1e}")
    return optimizer


# ==========================================
# 4. 数据集拆分与评估（同 train.py）
# ==========================================
def get_train_val_test_split(data_dir, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    按「原始样本」为单位分层划分（默认 8:1:1），读取 augment_manifest.json
    确保同一原始样本的所有增强版本进入同一集合，防止数据泄漏。
    测试集文件路径写入 test_split.json，供 classify.py 独立评估。

    augment_manifest.json 不存在时，退化为按文件粒度的普通划分。
    """
    manifest_path = os.path.join(data_dir, "augment_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = None
        print(">>> [WARN] augment_manifest.json 未找到，将按文件粒度划分（不含数据泄漏防护）")

    def _gather_with_augs(cat_prefix):
        """
        以原始样本为单位收集该类别的所有文件（原始 + 增强）。
        返回: [(original_path, [original_path, aug1, aug2, ...]), ...]
        无 manifest 时每个文件单独作为一组。
        """
        cat_pattern = cat_prefix.replace("\\", "/")  # FD or OF

        if manifest is not None:
            # 从 manifest 中筛选该类别的原始样本，按路径排序保证确定性
            cat_originals = sorted(
                [k for k in manifest if os.path.basename(k).startswith(cat_pattern + "_")],
                key=lambda p: os.path.basename(p),
            )
            groups = []
            for orig in cat_originals:
                # 原始文件 + 所有增强文件（只保留实际存在的）
                all_files = [orig]
                for aug in manifest.get(orig, []):
                    if os.path.exists(aug):
                        all_files.append(aug)
                groups.append((orig, all_files))
            return groups
        else:
            # 退化：每个文件单独一组
            files = sorted(glob.glob(os.path.join(data_dir, cat_prefix, "*.nii.gz")))
            return [(f, [f]) for f in files]

    fd_groups = _gather_with_augs("FD")
    of_groups = _gather_with_augs("OF")

    fd_total = sum(len(g[1]) for g in fd_groups)
    of_total = sum(len(g[1]) for g in of_groups)
    print(f">>> 数据集: FD={len(fd_groups)} 原始样本 ({fd_total} 文件), "
          f"OF={len(of_groups)} 原始样本 ({of_total} 文件)")

    # 按原始样本为单位 shuffle + 切分
    random.seed(random_seed)
    random.shuffle(fd_groups)
    random.shuffle(of_groups)

    fd_val_n = max(1, int(len(fd_groups) * val_ratio))
    fd_test_n = max(1, int(len(fd_groups) * test_ratio))
    fd_train_n = len(fd_groups) - fd_val_n - fd_test_n

    of_val_n = max(1, int(len(of_groups) * val_ratio))
    of_test_n = max(1, int(len(of_groups) * test_ratio))
    of_train_n = len(of_groups) - of_val_n - of_test_n

    def _expand_groups(groups, label):
        """将原始样本组展开为 [{image, label}, ...]"""
        samples = []
        for _orig_path, files in groups:
            for f in files:
                samples.append({"image": f, "label": label})
        return samples

    fd_train_groups = fd_groups[:fd_train_n]
    fd_val_groups   = fd_groups[fd_train_n:fd_train_n + fd_val_n]
    fd_test_groups  = fd_groups[fd_train_n + fd_val_n:]

    of_train_groups = of_groups[:of_train_n]
    of_val_groups   = of_groups[of_train_n:of_train_n + of_val_n]
    of_test_groups  = of_groups[of_train_n + of_val_n:]

    train_files = _expand_groups(fd_train_groups, 0) + _expand_groups(of_train_groups, 1)
    val_files   = _expand_groups(fd_val_groups, 0)   + _expand_groups(of_val_groups, 1)
    test_files  = _expand_groups(fd_test_groups, 0)  + _expand_groups(of_test_groups, 1)

    random.shuffle(train_files)
    random.shuffle(val_files)

    # 保存测试集文件列表供 classify.py 使用
    split_path = os.path.join(os.path.dirname(__file__), "test_split.json")
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(test_files, fh, ensure_ascii=False, indent=2)
    print(f">>> 测试集列表已保存: {split_path}")

    # 日志
    def _count(x, lbl): return sum(1 for d in x if d["label"] == lbl)
    print(f">>> 训练集: {len(train_files)} 例 (FD={_count(train_files,0)}, OF={_count(train_files,1)})")
    print(f">>> 验证集: {len(val_files)} 例   (FD={_count(val_files,0)}, OF={_count(val_files,1)})")
    print(f">>> 测试集: {len(test_files)} 例  (FD={_count(test_files,0)}, OF={_count(test_files,1)})")

    return train_files, val_files, test_files


@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}
    use_amp = device.type == "cuda"

    for batch_data in test_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        with autocast(device.type, enabled=use_amp):
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for lbl, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            class_total[int(lbl)] += 1
            if lbl == pred:
                class_correct[int(lbl)] += 1

    overall_acc = 100.0 * correct / total if total > 0 else 0.0
    fd_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
    of_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0
    return overall_acc, fd_acc, of_acc, correct, total


# ==========================================
# 5. 主训练流程（含迁移学习策略）
# ==========================================
def main():
    # 固定随机种子，保证实验可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    image_size = args.image_size

    # ---- 数据准备：8:1:1 三路划分（按原始样本分组，防泄漏） ----
    train_files, val_files, test_files = get_train_val_test_split(args.data_dir)
    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),  # 矢状轴（左右）翻转，解剖学合理
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
    ])

    train_loader = DataLoader(
        Dataset(data=train_files, transform=train_transforms),
        batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        Dataset(data=val_files, transform=val_transforms),
        batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    # 测试集不参与训练，已保存到 test_split.json 供 classify.py 独立评估

    # ---- 模型构建 ----
    base_vsnet = VSNet(img_size=image_size, in_channels=1, out_channels=3, training=False)
    model = VSNetClassifier(base_vsnet, num_classes=2).to(device)

    # ---- 迁移学习: 加载预训练权重 ----
    if not args.no_pretrain:
        pretrained_loaded = load_pretrained_weights(model, device)
        if pretrained_loaded:
            # 阶段 1: 冻结 backbone，只训练分类头
            freeze_backbone(model, freeze=True)
            print("[阶段1] backbone 已冻结，仅训练分类头")
    else:
        pretrained_loaded = False
        print("[对比实验] 不使用预训练权重，随机初始化训练")

    # ---- 类平衡权重（处理 FD/OF 样本不均衡） ----
    fd_train = sum(1 for s in train_files if s["label"] == 0)
    of_train = sum(1 for s in train_files if s["label"] == 1)
    total_train = fd_train + of_train
    class_weight = torch.tensor([
        total_train / (2 * fd_train) if fd_train > 0 else 1.0,
        total_train / (2 * of_train) if of_train > 0 else 1.0,
    ]).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weight)
    print(f">>> 类平衡权重: FD={class_weight[0]:.3f}, OF={class_weight[1]:.3f}")

    scaler = GradScaler(device.type, enabled=use_amp)

    best_val_acc = 0.0

    for epoch in range(args.epochs):
        # ---- 渐进式解冻 ----
        if pretrained_loaded and epoch == args.freeze_epochs:
            freeze_backbone(model, freeze=False)
            print("[阶段2] backbone 解冻，全模型微调")
            # 解冻后降低学习率，避免破坏预训练权重（已内置 bias/norm weight_decay=0）
            optimizer = setup_differential_lr(model, base_lr=args.lr * 0.5)
        elif epoch == 0 or (not pretrained_loaded and epoch == 0):
            optimizer = build_optimizer(model, lr=args.lr, weight_decay=1e-4)

        # ---- 训练 ----
        model.train()
        epoch_loss = 0
        step = 0
        grad_accum_steps = max(1, 4 // args.batch_size)  # 目标有效 batch_size ≈ 4

        optimizer.zero_grad()
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            with autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            # 梯度累加：将 loss 按累加步数缩放，兼容 batch_size=1 时的降级方案
            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps  # 恢复原始 loss 用于日志

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        # 处理 epoch 末尾剩余的累积梯度（总步数不被 grad_accum_steps 整除时）
        if step % grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss = epoch_loss / step  # 归一化为单步均值

        # ---- 评估（验证集） ----
        val_acc, fd_acc, of_acc, _, _ = evaluate(model, val_loader, device)
        phase = "冻结" if (pretrained_loaded and epoch < args.freeze_epochs) else "微调"
        print(f"Epoch [{epoch+1}/{args.epochs}][{phase}] "
              f"Loss: {epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.2f}% (FD:{fd_acc:.2f}% OF:{of_acc:.2f}%)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)
            print(f"  >>> 保存最优模型 (Val Acc={best_val_acc:.2f}%)")

    # ---- 最终评估（验证集） ----
    print(f"\n{'='*60}")
    print(f"  迁移学习训练完成！最优验证准确率: {best_val_acc:.2f}%")
    print(f"  测试集未参与训练，请用 classify.py 评估:")
    print(f"    python classify.py --model {args.save_path} --test_split test_split.json")
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path, map_location=device))
    val_acc, fd_acc, of_acc, correct, total = evaluate(model, val_loader, device)
    print(f"  最终验证集: {val_acc:.2f}% (FD:{fd_acc:.2f}% OF:{of_acc:.2f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
