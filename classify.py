"""
分类推理脚本 —— 用训练好的模型对新样本进行分类，并报告准确度。

用法：
  # 按类别目录组织（自动检测标签并计算准确率）
  python classify.py --model best_model_tl.pth --data_dir F:/path/to/new_data

  # 单文件推理
  python classify.py --model best_model_tl.pth --single F:/path/to/sample.nii.gz

  # 指定模型和数据目录
  python classify.py --model best_model.pth --data_dir F:/path/to/data --image_size 96

输出：
  - 每个样本的预测类别、标签（如有）、置信度
  - 混淆矩阵（如有标签）
  - 整体准确率 + 各类别准确率（如有标签）
"""
import os
import glob
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized
)
from monai.data import Dataset, DataLoader

from VSNet import VSNet

# ==========================================
# 命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="VSNet 分类推理")
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--data_dir", type=str, default=None,
                   help="待分类数据目录（按 FD/OF 子目录组织时有标签，否则纯推理）")
group.add_argument("--single", type=str, default=None,
                   help="单文件推理：指定一个 .nii.gz 文件路径")
group.add_argument("--test_split", type=str, default=None,
                   help="从训练时保存的 test_split.json 加载测试集并评估")
parser.add_argument("--model", type=str, default="best_model_tl.pth",
                    help="模型权重文件路径")
parser.add_argument("--image_size", type=int, default=96, help="输入尺寸（需与训练时一致）")
parser.add_argument("--device", type=str, default="cuda",
                    help="推理设备 (cuda / cpu)")
args = parser.parse_args()


# ==========================================
# 1. 分类器（与训练时完全一致）
# ==========================================
class VSNetClassifier(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super().__init__()
        self.backbone = original_model
        in_features = 16 * 12
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
# 2. 模型加载
# ==========================================
def load_model(model_path, image_size, device):
    """加载训练好的模型权重"""
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        sys.exit(1)

    base_vsnet = VSNet(img_size=image_size, in_channels=1, out_channels=3, training=False)
    model = VSNetClassifier(base_vsnet, num_classes=2).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"✅ 已加载模型: {model_path}")
    return model


# ==========================================
# 3. 数据加载（自动检测是否有标签）
# ==========================================
CLASS_NAMES = {0: "FD", 1: "OF"}

def build_samples(data_dir):
    """
    扫描数据目录，构建样本列表。
    若存在 FD/OF 子目录 → 有标签；否则 → 纯推理（无标签）。
    返回 (samples, has_label)
        samples = [{"image": path, "label": 0_or_1_or_-1}, ...]
    """
    fd_dir = os.path.join(data_dir, "FD")
    of_dir = os.path.join(data_dir, "OF")
    has_label = os.path.isdir(fd_dir) or os.path.isdir(of_dir)

    samples = []

    if has_label:
        # 按 FD/OF 目录组织
        if os.path.isdir(fd_dir):
            for f in sorted(glob.glob(os.path.join(fd_dir, "*.nii.gz"))):
                samples.append({"image": f, "label": 0})
        if os.path.isdir(of_dir):
            for f in sorted(glob.glob(os.path.join(of_dir, "*.nii.gz"))):
                samples.append({"image": f, "label": 1})
    else:
        # 无标签目录，直接扫描所有 .nii.gz
        for f in sorted(glob.glob(os.path.join(data_dir, "*.nii.gz"))):
            samples.append({"image": f, "label": -1})
        # 也可能在子目录中
        for f in sorted(glob.glob(os.path.join(data_dir, "*", "*.nii.gz"))):
            samples.append({"image": f, "label": -1})

    if not samples:
        print(f"❌ 在 {data_dir} 中未找到 .nii.gz 文件")
        sys.exit(1)

    return samples, has_label


def load_test_split(split_path):
    """
    从训练时保存的 test_split.json 加载测试集样本列表。
    返回 (samples, has_label=True)
    """
    import json as _json

    if not os.path.exists(split_path):
        print(f"❌ 测试集文件不存在: {split_path}")
        sys.exit(1)

    with open(split_path, "r", encoding="utf-8") as fh:
        samples = _json.load(fh)

    # 验证格式
    for s in samples:
        if "image" not in s or "label" not in s:
            print(f"❌ test_split.json 格式错误，缺少 image/label 字段")
            sys.exit(1)
        if not os.path.exists(s["image"]):
            print(f"⚠️  测试集文件不存在: {s['image']}")

    print(f"✅ 已加载测试集: {len(samples)} 例")
    return samples, True  # 测试集必定有标签


# ==========================================
# 4. 推理主流程
# ==========================================
@torch.no_grad()
def classify(model, samples, image_size, device):
    """
    逐样本推理，返回每个样本的预测结果。
    """
    # 预处理（无增强）
    transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
    ])

    results = []

    for sample in samples:
        try:
            data = transforms({"image": sample["image"]})
            inp = data["image"].unsqueeze(0).to(device)  # [1, C, D, H, W]

            output = model(inp)
            probs = torch.softmax(output, dim=1).squeeze(0)
            pred = int(torch.argmax(probs).item())
            confidence = float(probs[pred].item())

            results.append({
                "path": sample["image"],
                "filename": os.path.basename(sample["image"]),
                "label": sample["label"],           # -1 表示无标签
                "pred": pred,
                "pred_name": CLASS_NAMES[pred],
                "confidence": confidence,
                "probs": probs.cpu().numpy(),
            })
        except Exception as e:
            print(f"  ⚠️  跳过 {sample['image']}: {e}")

    return results


# ==========================================
# 5. 结果输出
# ==========================================
def print_results(results, has_label):
    """格式化输出推理结果"""

    # ---- 逐样本表格 ----
    print(f"\n{'='*80}")
    if has_label:
        print(f"{'文件名':<30} {'真实':>6} {'预测':>6} {'置信度':>10} {'结果':>8}")
    else:
        print(f"{'文件名':<35} {'预测':>6} {'置信度':>10}")
    print(f"{'-'*80}")

    correct = 0
    total = 0
    class_correct = {0: 0, 1: 0}
    class_total = {0: 0, 1: 0}

    for r in results:
        conf_str = f"{r['confidence']*100:.1f}%"
        if has_label:
            label_name = CLASS_NAMES.get(r["label"], "?")
            is_correct = "✓" if r["label"] == r["pred"] else "✗"
            print(f"{r['filename']:<30} {label_name:>6} {r['pred_name']:>6} {conf_str:>10} {is_correct:>8}")

            total += 1
            if r["label"] == r["pred"]:
                correct += 1
            class_total[r["label"]] += 1
            if r["label"] == r["pred"]:
                class_correct[r["label"]] += 1
        else:
            print(f"{r['filename']:<35} {r['pred_name']:>6} {conf_str:>10}")

    # ---- 汇总统计（仅在有标签时） ----
    if has_label and total > 0:
        overall_acc = 100.0 * correct / total
        fd_acc = 100.0 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0.0
        of_acc = 100.0 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0.0

        print(f"\n{'='*80}")
        print(f"  评估汇总")
        print(f"{'='*80}")
        print(f"  样本总数:     {total}")
        print(f"  正确预测:     {correct}")
        print(f"  错误预测:     {total - correct}")
        print(f"  ────────────────────────────")
        print(f"  整体准确率:   {overall_acc:.2f}%")
        print(f"  FD 准确率:    {fd_acc:.2f}%  ({class_correct[0]}/{class_total[0]})")
        print(f"  OF 准确率:    {of_acc:.2f}%  ({class_correct[1]}/{class_total[1]})")

        # ---- 混淆矩阵 ----
        print(f"\n  混淆矩阵:")
        print(f"                     预测")
        print(f"                FD        OF")
        # TP/FP/FN/TN
        tp_of = class_correct[1]
        fn_of = class_total[1] - class_correct[1]
        tn_fd = class_correct[0]
        fp_fd = class_total[0] - class_correct[0]
        print(f"  真实  FD    {tn_fd:>5}      {fp_fd:>5}")
        print(f"        OF    {fn_of:>5}      {tp_of:>5}")

        print(f"{'='*80}")

    elif not has_label:
        # 纯推理模式：统计预测分布
        fd_count = sum(1 for r in results if r["pred"] == 0)
        of_count = sum(1 for r in results if r["pred"] == 1)
        print(f"\n  预测分布: FD={fd_count}, OF={of_count} (共 {len(results)} 例)")
        print(f"{'='*80}")


# ==========================================
# 6. 主入口
# ==========================================
def main():
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f">>> 推理设备: {device}")

    # ---- 加载模型 ----
    model = load_model(args.model, args.image_size, device)

    # ---- 构建样本列表 ----
    if args.single:
        samples = [{"image": args.single, "label": -1}]
        has_label = False
    elif args.test_split:
        samples, has_label = load_test_split(args.test_split)
    else:
        samples, has_label = build_samples(args.data_dir)

    if has_label:
        n_fd = sum(1 for s in samples if s["label"] == 0)
        n_of = sum(1 for s in samples if s["label"] == 1)
        print(f">>> 待分类样本: FD={n_fd} 例, OF={n_of} 例 (共 {len(samples)} 例)")
    else:
        print(f">>> 待分类样本: {len(samples)} 例 (无标签)")

    # ---- 推理 ----
    results = classify(model, samples, args.image_size, device)

    # ---- 输出 ----
    print_results(results, has_label)


if __name__ == "__main__":
    main()
