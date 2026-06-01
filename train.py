import os
import glob
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
# 2. 构建数据集字典列表
# ==========================================
def get_data_dicts(data_dir):
    fd_files = glob.glob(os.path.join(data_dir, "FD", "*.nii.gz"))
    of_files = glob.glob(os.path.join(data_dir, "OF", "*.nii.gz"))
    
    data_dicts = []
    # 设定：FD 类别标签为 0，OF 类别标签为 1
    for f in fd_files:
        data_dicts.append({"image": f, "label": 0})
    for f in of_files:
        data_dicts.append({"image": f, "label": 1})
    return data_dicts

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    # 路径配置
    data_dir = r"F:\python\3DCT_Classification\dataset\NIfTI_Data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 改回原生 96 尺寸，防止模型内部 einops 报尺寸错
    image_size = 96 
    
    # 获取数据集
    data_dicts = get_data_dicts(data_dir)
    if len(data_dicts) == 0:
        print("❌ 错误：未在指定目录下找到任何转换好的 .nii.gz 文件，请先确认预处理是否成功。")
        return
        
    train_files = data_dicts 

    # MONAI 数据预处理与强数据增强
    train_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader"), # 强制指定 ITKReader
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityd(keys=["image"]), 
        Resized(keys=["image"], spatial_size=(image_size, image_size, image_size)), 
        RandRotate90d(keys=["image"], prob=0.5, spatial_axes=(0, 2)),
        RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
    ])
    
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)

    # 实例化模型
    base_vsnet = VSNet(img_size=image_size, in_channels=1, out_channels=3, training=False)
    model = VSNetClassifier(base_vsnet, num_classes=2).to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    
    # 升级：指定设备为 'cuda'
    scaler = GradScaler('cuda')

    print(f">>> Device detected: {device}. Starting training loop...")
    epochs = 40
    for epoch in range(epochs):
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
            
        print(f"Epoch [{epoch+1}/{epochs}] -> Average Training Loss: {epoch_loss/step:.4f}")
    
    print("\n>>> Training process finished successfully!")

if __name__ == "__main__":
    main()