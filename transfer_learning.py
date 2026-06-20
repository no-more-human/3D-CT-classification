"""
迁移学习训练脚本 —— 使用 MONAI 预训练模型初始化 VSNet，解决小样本过拟合问题。
已彻底重构：支持 VSNet 纯分类模式（mode="classification"），移除了冗余的包装器与 Decoder。
"""
import os
import glob
import json
import random
import copy

import argparse
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, RandRotate90d, RandFlipd,
    RandAffined, RandGaussianNoised,
)
from monai.data import Dataset

from VSNet import VSNet

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="VSNet 迁移学习训练")
parser.add_argument("--lr", type=float, default=2e-4, help="初始学习率（分类头）")
parser.add_argument("--epochs", type=int, default=40, help="训练总轮数")
parser.add_argument("--batch_size", type=int, default=4, help="批次大小")
parser.add_argument("--data_dir", type=str, default=r"F:\python\3DCT_Classification\dataset\NIfTI_Data", help="数据集路径")
parser.add_argument("--image_size", type=int, default=96, help="输入尺寸")
parser.add_argument("--no_pretrain", action="store_true", help="不使用预训练权重")
parser.add_argument("--save_path", type=str, default="best_model_tl.pth", help="模型保存路径")
parser.add_argument("--dropout", type=float, default=0.3, help="CSA/SSA/FC 的 dropout 率")
parser.add_argument("--weight_decay", type=float, default=1e-2, help="AdamW 权重衰减系数")
parser.add_argument("--backbone_lr_factor", type=float, default=0.01, help="CNN backbone 学习率缩放因子")
parser.add_argument("--swin_lr_factor", type=float, default=0.1, help="Swin Transformer 学习率缩放因子")
parser.add_argument("--kfold", type=int, default=5, help="K-Fold 交叉验证折数（设为 0 或 1 则为标准单次划分模式）")
parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例（仅非 K-Fold 模式）")
parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例（仅非 K-Fold 模式）")
parser.add_argument("--log_file", type=str, default="training_log.json", help="训练指标日志文件路径")
args = parser.parse_args()


# ==========================================
# 1. 预训练权重加载（MONAI Model Zoo）
# ==========================================
def load_pretrained_weights(model, device):
    pretrained_loaded = False
    model_pt = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "pretrained_models", "swin_unetr_btcv_segmentation", "models", "model.pt",
    )
    try:
        if not os.path.exists(model_pt):
            print("[迁移学习] 本地未找到预训练权重，尝试从 MONAI Model Zoo 下载...")
            from monai.bundle import load
            load(
                name="swin_unetr_btcv_segmentation",
                bundle_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models"),
                source="github",
            )
            if not os.path.exists(model_pt):
                raise FileNotFoundError(f"下载后仍未找到: {model_pt}")

        print("[迁移学习] 加载预训练 SwinUNETR 权重...")
        state_dict = torch.load(model_pt, map_location="cpu")

        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]

        mapped_count = _map_swin_weights(model, state_dict)
        
        if mapped_count > 0:
            pretrained_loaded = True
            print(f"[迁移学习] [OK] 成功加载并映射了 {mapped_count} 个层的预训练权重")
        else:
            print("[迁移学习] [WARN] 匹配到的权重层数为 0，加载失败。")
            
    except Exception as e:
        print(f"[迁移学习] 预训练权重加载失败: {e}")

    if not pretrained_loaded:
        print("[迁移学习] [WARN] 启用退化策略：Kaiming 初始化整个网络")
        _apply_kaiming_init(model)
        warnings.warn("预训练权重加载失败，将在全随机初始化下训练。")

    return pretrained_loaded


def _map_swin_weights(model, pretrained_dict):
    if hasattr(pretrained_dict, "state_dict"):
        pretrained_dict = pretrained_dict.state_dict()

    # 1. 粗略清洗前缀
    clean_pretrained_dict = {}
    for k, v in pretrained_dict.items():
        k_clean = k.replace("module.", "").replace("network.", "").replace("net.", "")
        clean_pretrained_dict[k_clean] = v

    model_dict = model.state_dict()
    mapped = 0

    # 统计 VSNet 中 swintransformer 一共有多少层
    target_keys = [k for k in model_dict if "swintransformer" in k]
    
    # 2. 核心逻辑：基于形状和后缀的暴力模糊匹配
    for k_model in target_keys:
        # 提取模型层的基本名称，例如 'blocks.0.norm1.weight'
        suffix = k_model.split("swintransformer.")[-1]
        target_shape = model_dict[k_model].shape
        
        matched_key = None
        # 遍历预训练字典，寻找后缀相同且形状一致的层
        for pt_key, pt_tensor in clean_pretrained_dict.items():
            if pt_key.endswith(suffix) and pt_tensor.shape == target_shape:
                matched_key = pt_key
                break
        
        # 特殊处理相对位置编码表 (截断适配)
        if matched_key is None and "relative_position_bias_table" in k_model:
            for pt_key, pt_tensor in clean_pretrained_dict.items():
                if pt_key.endswith(suffix) and len(pt_tensor.shape) == 2 and len(target_shape) == 2:
                    if pt_tensor.shape[0] == target_shape[0]: 
                        model_dict[k_model] = pt_tensor[:, :target_shape[1]].contiguous()
                        mapped += 1
                        matched_key = "special_matched"
                        break

        # 如果找到了，就赋给当前模型
        if matched_key and matched_key != "special_matched":
            model_dict[k_model] = clean_pretrained_dict[matched_key]
            mapped += 1

    model.load_state_dict(model_dict, strict=False)
    print(f"[迁移学习]   [OK] 成功使用暴力匹配映射了 {mapped} 个层的预训练权重 （VSNet 共 {len(target_keys)} 层）")
    
    return mapped

def _apply_kaiming_init(model):
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
# 2. 优化器与参数拆分
# ==========================================
def _split_params_by_weight_decay(named_params, lr, weight_decay):
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

def selective_freeze(model, freeze_cnn=True):
    cnn_prefixes = ("resuetencoder", "pool", "gate") 
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in cnn_prefixes):
            param.requires_grad = not freeze_cnn
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[选择性冻结] CNN encoder {'已冻结' if freeze_cnn else '已解冻'}，"
          f"可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

def setup_selective_optimizer(model, base_lr, backbone_factor, swin_factor, weight_decay):
    cnn_prefixes = ("resuetencoder", "pool", "gate")
    swin_prefixes = ("swintransformer", "CSA", "SSA")

    cnn_params, swin_params, head_params = [], [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(name.startswith(p) for p in cnn_prefixes):
            cnn_params.append((name, param))
        elif any(name.startswith(p) for p in swin_prefixes):
            swin_params.append((name, param))
        else:
            head_params.append((name, param))

    all_groups = []
    if cnn_params:
        all_groups.extend(_split_params_by_weight_decay(cnn_params, base_lr * backbone_factor, weight_decay))
    if swin_params:
        all_groups.extend(_split_params_by_weight_decay(swin_params, base_lr * swin_factor, weight_decay))
    if head_params:
        all_groups.extend(_split_params_by_weight_decay(head_params, base_lr, weight_decay))

    optimizer = torch.optim.AdamW(all_groups)
    n_cnn = sum(p.numel() for _, p in cnn_params)
    n_swin = sum(p.numel() for _, p in swin_params)
    n_head = sum(p.numel() for _, p in head_params)
    print(f"[差分学习率] CNN: {base_lr * backbone_factor:.1e} ({n_cnn:,} params) | "
          f"Swin+Attn: {base_lr * swin_factor:.1e} ({n_swin:,} params) | "
          f"Head: {base_lr:.1e} ({n_head:,} params) | weight_decay={weight_decay}")
    return optimizer


# ==========================================
# 3. 数据集划分函数
# ==========================================
def get_train_val_test_split(data_dir, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    manifest_path = os.path.join(data_dir, "augment_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = None
        print(">>> [WARN] augment_manifest.json 未找到，将按文件粒度划分（不含数据泄漏防护）")

    def _gather_with_augs(cat_prefix):
        cat_pattern = cat_prefix.replace("\\", "/")
        if manifest is not None:
            cat_originals = sorted(
                [k for k in manifest if os.path.basename(k).startswith(cat_pattern + "_")],
                key=lambda p: os.path.basename(p),
            )
            groups = []
            for orig in cat_originals:
                all_files = [orig]
                for aug in manifest.get(orig, []):
                    if os.path.exists(aug):
                        all_files.append(aug)
                groups.append((orig, all_files))
            return groups
        else:
            files = sorted(glob.glob(os.path.join(data_dir, cat_prefix, "*.nii.gz")))
            return [(f, [f]) for f in files]

    fd_groups = _gather_with_augs("FD")
    of_groups = _gather_with_augs("OF")

    fd_total = sum(len(g[1]) for g in fd_groups)
    of_total = sum(len(g[1]) for g in of_groups)
    print(f">>> 数据集: FD={len(fd_groups)} 原始样本 ({fd_total} 文件), OF={len(of_groups)} 原始样本 ({of_total} 文件)")

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

    split_path = os.path.join(os.path.dirname(__file__), "test_split.json")
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(test_files, fh, ensure_ascii=False, indent=2)
    print(f">>> 测试集列表已保存: {split_path}")

    def _count(x, lbl): return sum(1 for d in x if d["label"] == lbl)
    print(f">>> 训练集: {len(train_files)} 例 (FD={_count(train_files,0)}, OF={_count(train_files,1)})")
    print(f">>> 验证集: {len(val_files)} 例   (FD={_count(val_files,0)}, OF={_count(val_files,1)})")
    print(f">>> 测试集: {len(test_files)} 例  (FD={_count(test_files,0)}, OF={_count(test_files,1)})")

    return train_files, val_files, test_files


def get_kfold_splits_with_test(data_dir, n_folds=5, test_ratio=0.1, random_seed=42):
    manifest_path = os.path.join(data_dir, "augment_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = None
        print(">>> [WARN] augment_manifest.json 未找到，将按文件粒度划分（不含数据泄漏防护）")

    def _gather_with_augs(cat_prefix):
        cat_pattern = cat_prefix.replace("\\", "/")
        if manifest is not None:
            cat_originals = sorted(
                [k for k in manifest if os.path.basename(k).startswith(cat_pattern + "_")],
                key=lambda p: os.path.basename(p),
            )
            groups = []
            for orig in cat_originals:
                all_files = [orig]
                for aug in manifest.get(orig, []):
                    if os.path.exists(aug):
                        all_files.append(aug)
                groups.append((orig, all_files))
            return groups
        else:
            files = sorted(glob.glob(os.path.join(data_dir, cat_prefix, "*.nii.gz")))
            return [(f, [f]) for f in files]

    fd_groups = _gather_with_augs("FD")
    of_groups = _gather_with_augs("OF")

    fd_total = sum(len(g[1]) for g in fd_groups)
    of_total = sum(len(g[1]) for g in of_groups)
    print(f">>> 数据集: FD={len(fd_groups)} 原始样本 ({fd_total} 文件), OF={len(of_groups)} 原始样本 ({of_total} 文件)")

    random.seed(random_seed)
    random.shuffle(fd_groups)
    random.shuffle(of_groups)

    fd_test_n = max(1, int(len(fd_groups) * test_ratio))
    of_test_n = max(1, int(len(of_groups) * test_ratio))

    fd_test_groups = fd_groups[:fd_test_n]
    fd_kfold_groups = fd_groups[fd_test_n:]

    of_test_groups = of_groups[:of_test_n]
    of_kfold_groups = of_groups[of_test_n:]

    def _expand_groups(groups, label):
        samples = []
        for _orig_path, files in groups:
            for f in files:
                samples.append({"image": f, "label": label})
        return samples

    test_files = _expand_groups(fd_test_groups, 0) + _expand_groups(of_test_groups, 1)

    split_path = os.path.join(os.path.dirname(__file__), "test_split.json")
    with open(split_path, "w", encoding="utf-8") as fh:
        json.dump(test_files, fh, ensure_ascii=False, indent=2)
    print(f">>> 测试集列表已保存: {split_path}")

    def _count(x, lbl): return sum(1 for d in x if d["label"] == lbl)
    print(f">>> 留出测试集: {len(test_files)} 例 (FD文件={_count(test_files,0)}, OF文件={_count(test_files,1)})")

    folds = []
    fd_fold_size = max(1, len(fd_kfold_groups) // n_folds)
    of_fold_size = max(1, len(of_kfold_groups) // n_folds)

    for fold in range(n_folds):
        fd_val_start = fold * fd_fold_size
        fd_val_end = min((fold + 1) * fd_fold_size, len(fd_kfold_groups))

        of_val_start = fold * of_fold_size
        of_val_end = min((fold + 1) * of_fold_size, len(of_kfold_groups))

        fd_val_groups = fd_kfold_groups[fd_val_start:fd_val_end]
        fd_train_groups = fd_kfold_groups[:fd_val_start] + fd_kfold_groups[fd_val_end:]

        of_val_groups = of_kfold_groups[of_val_start:of_val_end]
        of_train_groups = of_kfold_groups[:of_val_start] + of_kfold_groups[of_val_end:]

        train_files = _expand_groups(fd_train_groups, 0) + _expand_groups(of_train_groups, 1)
        val_files   = _expand_groups(fd_val_groups, 0)   + _expand_groups(of_val_groups, 1)

        random.shuffle(train_files)

        print(f">>> Fold {fold+1}/{n_folds}: 训练 {len(train_files)} 例 | 验证 {len(val_files)} 例")
        folds.append((train_files, val_files))

    return folds, test_files


# ==========================================
# 4. 评估指标计算
# ==========================================
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
# 5. 单次训练核心控制逻辑
# ==========================================
def train_one_run(model, train_files, val_files, device, fold_name="", log_file=None):
    image_size = args.image_size
    use_amp = device.type == "cuda"

    # 训练日志
    log_entries = []
    if log_file is None:
        log_file = args.log_file

    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="NiBabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
        RandAffined(keys=["image"], prob=0.5, translate_range=5, scale_range=0.1, spatial_size=(image_size, image_size, image_size), mode="bilinear", padding_mode="border"),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.05),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
    ])
    val_transforms = Compose([
        LoadImaged(keys=["image"], reader="NiBabelReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
    ])

    train_loader = DataLoader(Dataset(data=train_files, transform=train_transforms), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(data=val_files, transform=val_transforms), batch_size=args.batch_size, shuffle=False, num_workers=0)

    fd_train = sum(1 for s in train_files if s["label"] == 0)
    of_train = sum(1 for s in train_files if s["label"] == 1)
    total_train = fd_train + of_train
    class_weight = torch.tensor([
        total_train / (2 * fd_train) if fd_train > 0 else 1.0,
        total_train / (2 * of_train) if of_train > 0 else 1.0,
    ]).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weight)
    print(f">>> 类平衡权重: FD={class_weight[0]:.3f}, OF={class_weight[1]:.3f}")

    selective_freeze(model, freeze_cnn=False)
    optimizer = setup_selective_optimizer(
        model, base_lr=args.lr, backbone_factor=args.backbone_lr_factor, swin_factor=args.swin_lr_factor, weight_decay=args.weight_decay,
    )

    scaler = GradScaler(device.type, enabled=use_amp)
    best_val_acc = 0.0
    best_state_dict = None
    prefix = f"[{fold_name}] " if fold_name else ""

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        step = 0
        grad_accum_steps = max(1, 8 // args.batch_size)

        optimizer.zero_grad()
        for batch_data in train_loader:
            step += 1
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            with autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * grad_accum_steps

            if step % grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if step % grad_accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss = epoch_loss / step
        val_acc, fd_acc, of_acc, _, _ = evaluate(model, val_loader, device)
        print(f"{prefix}Epoch [{epoch+1}/{args.epochs}] Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}% (FD:{fd_acc:.2f}% OF:{of_acc:.2f}%)")

        log_entries.append({
            "epoch": epoch + 1,
            "loss": round(epoch_loss, 6),
            "val_acc": round(val_acc, 2),
            "fd_acc": round(fd_acc, 2),
            "of_acc": round(of_acc, 2),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            print(f"  {prefix}>>> 发现更优模型，暂存权重 (Val Acc={best_val_acc:.2f}%)")

    # 保存训练日志
    import json as _json
    with open(log_file, "w", encoding="utf-8") as fh:
        _json.dump(log_entries, fh, ensure_ascii=False, indent=2)
    print(f"{prefix}>>> 训练日志已保存: {log_file}")

    return best_val_acc, best_state_dict


# ==========================================
# 6. 主训练流程入口
# ==========================================
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    print(f"{'='*60}")
    print(f"  VSNet 迁移学习训练（小样本/纯分类融合优化版）")
    print(f"  设备: {device} | AMP: {use_amp}")
    print(f"  Dropout: {args.dropout} | Weight Decay: {args.weight_decay}")
    print(f"{'='*60}")

    if args.kfold > 1:
        folds, test_files = get_kfold_splits_with_test(args.data_dir, n_folds=args.kfold, test_ratio=args.test_ratio)
        fold_accs = []
        best_fold_state = None
        best_fold_acc = 0.0
        best_fold_model = None

        # 初始化基础模型并尝试加载预训练权重
        print("\n>>> 开始初始化 K-Fold 基础模型结构...")
        base_model = VSNet(
            in_channels=1, 
            out_channels=3,             # 传参兼容原定义
            num_classes=2,              # 最终分类分类头输出为2 (FD, OF)
            mode="classification",       # 开启纯分类模式
            img_size=args.image_size, 
            drop_rate=args.dropout,
            training=True
        ).to(device)

        if not args.no_pretrain:
            load_pretrained_weights(base_model, device)
        else:
            print("[迁移学习] 跳过预训练权重加载，采用随机 Kaiming 初始化")
            _apply_kaiming_init(base_model)

        # 深拷贝干净的初始状态（含预训练权重），供每折开始前重置
        initial_state = copy.deepcopy(base_model.state_dict())

        for fold_idx, (train_files, val_files) in enumerate(folds):
            fold_name = f"Fold_{fold_idx+1}"
            print(f"\n==================== 开始训练: {fold_name} ====================")
            
            # 每一折都重新初始化干净的权重，防止数据污染
            base_model.load_state_dict(initial_state)

            fold_acc, fold_state = train_one_run(base_model, train_files, val_files, device, fold_name=fold_name)
            fold_accs.append(fold_acc)

            if fold_acc > best_fold_acc:
                best_fold_acc = fold_acc
                best_fold_state = fold_state
                best_fold_model = copy.deepcopy(base_model)

        print(f"\n  K-Fold 交叉验证完成 | 平均准确率: {np.mean(fold_accs):.2f}%")
        
        if best_fold_state is not None:
            torch.save(best_fold_state, args.save_path)
            print(f">>> 历史最优折权重已成功保存至: {args.save_path}")

        # ---- 独立测试集评估 ----
        print(f"\n{'='*60}")
        print(f"  独立测试集评估（K-Fold 全程未参与训练）")
        print(f"{'='*60}")
        if best_fold_state is not None and best_fold_model is not None:
            best_fold_model.load_state_dict(best_fold_state)
            val_transforms = Compose([
                LoadImaged(keys=["image"], reader="NiBabelReader"),
                EnsureChannelFirstd(keys=["image"]),
                ScaleIntensityd(keys=["image"]),
                Resized(keys=["image"], spatial_size=(args.image_size, args.image_size, args.image_size)),
            ])
            test_loader = DataLoader(Dataset(data=test_files, transform=val_transforms), batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_acc, test_fd_acc, test_of_acc, _, total = evaluate(best_fold_model, test_loader, device)
            print(f"  测试集总样本: {total} 例")
            print(f"  整体准确率: {test_acc:.2f}% | FD: {test_fd_acc:.2f}% | OF: {test_of_acc:.2f}%")
            print(f"{'='*60}")

    else:
        # ============ 原始标准单次划分模式 (完全对齐新分类逻辑) ============
        print("\n>>> 启动标准单次划分训练模式（Train/Val/Test 模式）...")
        train_files, val_files, test_files = get_train_val_test_split(args.data_dir, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

        model = VSNet(
            in_channels=1, 
            out_channels=3, 
            num_classes=2, 
            mode="classification", 
            img_size=args.image_size, 
            drop_rate=args.dropout,
            training=True
        ).to(device)

        if not args.no_pretrain:
            load_pretrained_weights(model, device)
        else:
            print("[迁移学习] 跳过预训练权重加载，采用随机 Kaiming 初始化")
            _apply_kaiming_init(model)

        best_val_acc, best_state_dict = train_one_run(model, train_files, val_files, device, fold_name="")

        if best_state_dict is not None:
            torch.save(best_state_dict, args.save_path)
            print(f">>> 最优模型权重已成功保存至: {args.save_path}")

        # ---- 测试集评估 ----
        print(f"\n{'='*60}")
        print(f"  独立测试集评估")
        print(f"{'='*60}")
        model.load_state_dict(best_state_dict)
        val_transforms = Compose([
            LoadImaged(keys=["image"], reader="NiBabelReader"),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityd(keys=["image"]),
            Resized(keys=["image"], spatial_size=(args.image_size, args.image_size, args.image_size)),
        ])
        test_loader = DataLoader(Dataset(data=test_files, transform=val_transforms), batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_acc, test_fd_acc, test_of_acc, _, total = evaluate(model, test_loader, device)
        print(f"  测试集总样本: {total} 例")
        print(f"  整体准确率: {test_acc:.2f}% | FD: {test_fd_acc:.2f}% | OF: {test_of_acc:.2f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()