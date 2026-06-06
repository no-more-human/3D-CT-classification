"""
离线数据增强脚本 —— 对现有 .nii.gz 样本生成增强副本，扩充数据集。

每个原始样本经过随机变换链产生 N 个增强版本，保存到同一类别目录下。
增强参数可调，每次运行会在控制台打印增强前后的样本数量统计。
"""
import os
import glob
import numpy as np
import torch
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    RandFlip,
    RandRotate,
    RandAffine,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
)


# ==========================================
# 可调参数
# ==========================================
DATA_DIR = r"F:\python\3DCT_Classification\dataset\NIfTI_Data"
AUGMENT_FACTOR = 5          # 每个原始样本生成的增强副本数
CATEGORIES = ["FD", "OF"]    # 分类类别


# ==========================================
# 3D CT 影像增强流水线
# ==========================================
def build_augment_pipeline():
    """
    构建 3D CT 增强流水线。
    每次调用随机施加不同组合的变换，产生多样性。
    """
    return Compose([
        # 1. 随机翻转：CT 沿矢状轴翻转不影响解剖学意义
        RandFlip(prob=0.5, spatial_axis=0),

        # 2. 随机旋转：±10° 小幅旋转，模拟患者摆位差异
        RandRotate(range_x=0.175, range_y=0.175, range_z=0.175,
                   prob=0.6, keep_size=True, mode="bilinear"),

        # 3. 随机仿射变换：剪切 + 缩放 + 平移
        RandAffine(
            prob=0.5,
            rotate_range=(0.1, 0.1, 0.1),
            shear_range=(0.05, 0.05, 0.05),
            scale_range=(0.05, 0.05, 0.05),
            translate_range=(5, 5, 5),
            padding_mode="border",
            mode="bilinear",
        ),

        # 4. 随机高斯噪声：模拟 CT 设备噪声
        RandGaussianNoise(prob=0.4, std=0.005, relative=True),

        # 5. 随机对比度调节：模拟不同窗宽窗位
        RandAdjustContrast(prob=0.3, gamma=(0.85, 1.15)),

        # 6. 随机高斯平滑：模拟不同重建核
        RandGaussianSmooth(prob=0.2, sigma_x=(0.25, 0.75),
                           sigma_y=(0.25, 0.75), sigma_z=(0.25, 0.75)),
    ])


def main():
    pipeline = build_augment_pipeline()

    total_original = 0
    total_augmented = 0

    for cat in CATEGORIES:
        cat_dir = os.path.join(DATA_DIR, cat)
        if not os.path.exists(cat_dir):
            print(f"⚠️  目录不存在，跳过: {cat_dir}")
            continue

        # 只拿原始文件（排除之前增强过的）
        files = sorted(glob.glob(os.path.join(cat_dir, "*.nii.gz")))
        originals = [f for f in files if "_aug" not in os.path.basename(f)]

        print(f"\n📂 {cat}: 原始 {len(originals)} 例 → 每例生成 {AUGMENT_FACTOR} 份增强样本")

        for src_path in originals:
            base_name = os.path.splitext(os.path.splitext(
                os.path.basename(src_path))[0])[0]  # e.g. "FD_1"

            try:
                sitk_img = sitk.ReadImage(src_path)
            except Exception as e:
                print(f"  ❌ 加载失败 {src_path}: {e}")
                continue

            # SimpleITK → numpy float32 → tensor [1, D, H, W]
            arr = sitk.GetArrayFromView(sitk_img).astype(np.float32)  # [D, H, W]
            img_tensor = torch.from_numpy(arr).unsqueeze(0)            # [1, D, H, W]

            for aug_idx in range(AUGMENT_FACTOR):
                # 应用增强流水线（每次随机不同）
                augmented = pipeline(img_tensor)                      # [1, D, H, W]

                # tensor → numpy → SimpleITK（继承原始 spacing/origin/direction）
                out_arr = augmented.squeeze(0).numpy().astype(np.float32)  # [D, H, W]
                out_img = sitk.GetImageFromArray(out_arr)
                out_img.CopyInformation(sitk_img)

                out_name = f"{base_name}_aug{aug_idx}.nii.gz"
                out_path = os.path.join(cat_dir, out_name)
                sitk.WriteImage(out_img, out_path)

            total_original += 1
            total_augmented += AUGMENT_FACTOR

    # ---- 汇总 ----
    print(f"\n{'='*50}")
    print(f"  增强完成！")
    print(f"  原始样本: {total_original} 例")
    print(f"  新增增强: {total_augmented} 例")
    print(f"  增强后总数: {total_original + total_augmented} 例")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
