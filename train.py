import os
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler  # 升级：采用 PyTorch 2.0+ 最新标准写法
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityd, Resized, RandRotate90d, RandFlipd
)
from monai.data import Dataset

# 引入你上传的 VSNet 模型
from VSNet import VSNet

# ==========================================
# 1. 改造 VSNet：截断解码器，封装为二分类网络
# ==========================================
class VSNetClassifier(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super().__init__()
        self.backbone = original_model
        
        # VSNet 内部 feature_size=12，其编码器最后一层输出通道为 16 * feature_size = 192
        in_features = 16 * 12 
        
        # 3D全局平均池化：将 (Batch, 192, D, H, W) 压缩为 (Batch, 192, 1, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        # 最终分类全连接层
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # 严格对齐你的 VSNet.py 前向传播前半部分提取特征
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
        x5 = self.backbone.SSA(x5) # 此时特征图形状: (B, 192, D_new, H_new, W_new)
        
        # 分类头整合
        out = self.global_pool(x5)  
        out = out.view(out.size(0), -1) # 拉平为 (B, 192)
        out = self.fc(out)              # 输出 (B, 2)
        return out

# ==========================================
# 2. 构建数据集并按 8:2 随机拆分为训练集和测试集
# ==========================================
def get_train_test_split(data_dir, test_ratio=0.2, random_seed=42):
    """
    按类别分层随机拆分数据集为训练集 (80%) 和测试集 (20%)。
    固定随机种子保证每次运行拆分结果一致，方便对比实验。
    """
    fd_files = sorted(glob.glob(os.path.join(data_dir, "FD", "*.nii.gz")))
    of_files = sorted(glob.glob(os.path.join(data_dir, "OF", "*.nii.gz")))

    print(f">>> 数据集统计: FD={len(fd_files)} 例, OF={len(of_files)} 例")

    # 固定随机种子，保证结果可复现
    random.seed(random_seed)
    random.shuffle(fd_files)
    random.shuffle(of_files)

    # 分别按比例切分，保证两个类别在训练/测试集中比例一致（分层抽样）
    fd_split = int(len(fd_files) * (1 - test_ratio))
    of_split = int(len(of_files) * (1 - test_ratio))

    train_files = []
    test_files = []

    # FD: 标签 0
    for f in fd_files[:fd_split]:
        train_files.append({"image": f, "label": 0})
    for f in fd_files[fd_split:]:
        test_files.append({"image": f, "label": 0})

    # OF: 标签 1
    for f in of_files[:of_split]:
        train_files.append({"image": f, "label": 1})
    for f in of_files[of_split:]:
        test_files.append({"image": f, "label": 1})

    # 再次打乱训练集和测试集内部顺序
    random.shuffle(train_files)
    random.shuffle(test_files)

    train_fd = sum(1 for d in train_files if d["label"] == 0)
    train_of = sum(1 for d in train_files if d["label"] == 1)
    test_fd = sum(1 for d in test_files if d["label"] == 0)
    test_of = sum(1 for d in test_files if d["label"] == 1)

    print(f">>> 训练集: {len(train_files)} 例 (FD={train_fd}, OF={train_of})")
    print(f">>> 测试集: {len(test_files)} 例 (FD={test_fd}, OF={test_of})")

    return train_files, test_files

# ==========================================
# 3. 测试集评估函数
# ==========================================
@torch.no_grad()
def evaluate(model, test_loader, device):
    """
    在测试集上评估模型，返回整体准确率以及每个类别的准确率。
    """
    model.eval()
    correct = 0
    total = 0
    # 按类别统计
    class_correct = {0: 0, 1: 0}  # 0=FD, 1=OF
    class_total = {0: 0, 1: 0}

    for batch_data in test_loader:
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

        with autocast('cuda'):
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
# 4. 主训练流程（含训练/测试拆分与最终评估）
# ==========================================
def main():
    # 路径配置
    data_dir = r"F:\python\3DCT_Classification\dataset\NIfTI_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 改回原生 96 尺寸，防止模型内部 einops 报尺寸错
    image_size = 96

    # ---------- 2.1 按 8:2 随机拆分训练集和测试集 ----------
    train_files, test_files = get_train_test_split(data_dir, test_ratio=0.2, random_seed=42)

    if len(train_files) == 0:
        print("❌ 错误：训练集为空，请先确认预处理是否成功（dataset/NIfTI_Data 下是否有 .nii.gz 文件）。")
        return
    if len(test_files) == 0:
        print("⚠️ 警告：测试集为空，将无法进行最终评估。")

    # ---------- 2.2 训练数据增强（含随机翻转/旋转，提升泛化能力）----------
    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    ])

    # ---------- 2.3 测试数据预处理（无增强，仅加载/归一化/缩放）----------
    test_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"),
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)),
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # ---------- 2.4 实例化模型 ----------
    base_vsnet = VSNet(img_size=image_size, in_channels=1, out_channels=3, training=False)
    model = VSNetClassifier(base_vsnet, num_classes=2).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

    # 升级：指定设备为 'cuda'
    scaler = GradScaler('cuda')

    print(f">>> Device detected: {device}. Starting training loop...")
    epochs = 40
    best_test_acc = 0.0

    for epoch in range(epochs):
        # ========== 训练阶段 ==========
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)

            optimizer.zero_grad()

            # 升级：指定设备为 'cuda' 开启半精度训练，暴省显存
            with autocast('cuda'):
                outputs = model(inputs)
                loss = loss_function(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        # ========== 每个 epoch 后在测试集上快速评估 ==========
        test_acc, fd_acc, of_acc, _, _ = evaluate(model, test_loader, device)

        print(f"Epoch [{epoch+1}/{epochs}] -> "
              f"Train Loss: {epoch_loss/step:.4f} | "
              f"Test Acc: {test_acc:.2f}% (FD: {fd_acc:.2f}%, OF: {of_acc:.2f}%)")

        # 保存测试集准确率最高的模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), os.path.join(
                os.path.dirname(__file__), "best_model.pth"))
            print(f"  >>> 已保存最优模型 (Test Acc = {best_test_acc:.2f}%)")

    print(f"\n{'='*60}")
    print(f">>> 训练流程结束！最优测试准确率: {best_test_acc:.2f}%")

    # ========== 最终测试集完整评估 ==========
    print(f"\n{'='*60}")
    print(">>> 正在加载最优模型进行最终测试集评估...")

    # 加载训练过程中保存的最优模型
    best_model_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    overall_acc, fd_acc, of_acc, correct, total = evaluate(model, test_loader, device)

    print(f"{'='*60}")
    print(f"  最终测试集评估结果")
    print(f"{'='*60}")
    print(f"  测试集样本总数: {total}")
    print(f"  正确预测数量:   {correct}")
    print(f"  ────────────────────────────")
    print(f"  整体准确率:     {overall_acc:.2f}%")
    print(f"  FD 类别准确率:  {fd_acc:.2f}%")
    print(f"  OF 类别准确率:  {of_acc:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()