"""
数据增强对比可视化 —— 提取原始样本与增强样本的 3 个正交中心切片并排对比。

用法：
  python plot_slices.py                          # 默认找 FD_1.nii.gz 和 FD_1_aug0.nii.gz
  python plot_slices.py --source FD_1 --aug_idx 5  # 指定原始样本和增强索引
  python plot_slices.py --source OF_5              # 指定 OF 类别样本

输出：plots/augment_comparison.png
"""

import os
import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==========================================
# 参数
# ==========================================
parser = argparse.ArgumentParser(description="数据增强对比可视化")
parser.add_argument("--data_dir", type=str,
                    default=r"F:\python\3DCT_Classification\dataset\NIfTI_Data",
                    help="数据目录")
parser.add_argument("--source", type=str, default="FD_1",
                    help="原始样本名（不含扩展名），如 FD_1")
parser.add_argument("--aug_idx", type=int, default=0,
                    help="增强索引 (0-9)")
args = parser.parse_args()

# ==========================================
# 核心函数
# ==========================================


def load_nifti(path: str) -> np.ndarray:
    """加载 .nii.gz → numpy [D, H, W] float32，并归一化到 [0,1]"""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)  # [D, H, W]
    # Min-Max 归一化到 [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    return arr


def get_middle_slice(arr_3d: np.ndarray):
    """返回 3 个正交面的中心切片 (axial, coronal, sagittal)"""
    d, h, w = arr_3d.shape
    axial = arr_3d[d // 2, :, :]       # 轴向 → 需要转置 [H, W]
    coronal = arr_3d[:, h // 2, :]     # 冠状 → [D, W] 但沿 Z 轴的冠状需翻转
    sagittal = arr_3d[:, :, w // 2]    # 矢状 → [D, H]
    return axial, coronal.T, sagittal.T  # coronal 和 sagittal 转置为更好的视觉效果


def plot_comparison(original_path: str, augmented_path: str, output_path: str):
    """3×3 网格：行=切面类型，左列=原始，右列=增强"""
    print(f">>> 原始: {original_path}")
    print(f">>> 增强: {augmented_path}")

    orig = load_nifti(original_path)
    aug = load_nifti(augmented_path)

    ax_orig, cor_orig, sag_orig = get_middle_slice(orig)
    ax_aug, cor_aug, sag_aug = get_middle_slice(aug)

    slice_labels = ["Axial (轴向)", "Coronal (冠状)", "Sagittal (矢状)"]
    orig_slices = [ax_orig, cor_orig, sag_orig]
    aug_slices = [ax_aug, cor_aug, sag_aug]

    fig, axes = plt.subplots(3, 2, figsize=(8, 12))
    fig.suptitle("数据增强前后对比 (中心切片)", fontsize=14, fontweight="bold", y=0.98)

    for row in range(3):
        # 左列：原始
        axes[row][0].imshow(orig_slices[row], cmap="gray", aspect="auto", origin="upper")
        axes[row][0].set_title(f"原始 — {slice_labels[row]}", fontsize=11)
        axes[row][0].axis("off")

        # 右列：增强
        axes[row][1].imshow(aug_slices[row], cmap="gray", aspect="auto", origin="upper")
        axes[row][1].set_title(f"增强 (aug{args.aug_idx}) — {slice_labels[row]}", fontsize=11)
        axes[row][1].axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] 已保存: {output_path}")


# ==========================================
# 主入口
# ==========================================
def main():
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots", "augment_comparison.png")

    # 解析 source 的类别
    cat = args.source.split("_")[0]  # "FD" or "OF"
    original_path = os.path.join(args.data_dir, cat, f"{args.source}.nii.gz")
    augmented_path = os.path.join(args.data_dir, cat, f"{args.source}_aug{args.aug_idx}.nii.gz")

    if not os.path.exists(original_path):
        print(f"[ERROR] 原始文件不存在: {original_path}")
        return
    if not os.path.exists(augmented_path):
        print(f"[ERROR] 增强文件不存在: {augmented_path}")
        return

    plot_comparison(original_path, augmented_path, output_path)


if __name__ == "__main__":
    main()
