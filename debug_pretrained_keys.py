"""诊断脚本：直接加载 model.pt 对比键名"""
import torch
from VSNet import VSNet

# 1. 直接加载 model.pt
print("=" * 60)
print("直接加载 SwinUNETR model.pt...")
state_dict = torch.load(
    r"F:\python\3DCT_Classification\pretrained_models\swin_unetr_btcv_segmentation\models\model.pt",
    map_location="cpu"
)

# Handle nested state_dict case
if "state_dict" in state_dict:
    print("  检测到嵌套，取 ['state_dict']")
    state_dict = state_dict["state_dict"]
elif "model" in state_dict:
    print("  检测到嵌套，取 ['model']")
    state_dict = state_dict["model"]

print(f"  总键数: {len(state_dict)}")

# 2. 打印所有 swinViT 相关的键
swinvit_keys = [k for k in state_dict if "swinViT" in k]
print(f"\n预训练模型中 swinViT 键: {len(swinvit_keys)} 个")
print("-" * 60)
for k in swinvit_keys:
    print(f"  {k}  →  shape={state_dict[k].shape}")

# 3. VSNet 中 swintransformer 键
print("\n" + "=" * 60)
print("VSNet 中 swintransformer 键:")
base_vsnet = VSNet(img_size=96, in_channels=1, out_channels=3, training=False)
model_dict = base_vsnet.state_dict()
swin_keys = [k for k in model_dict if "swintransformer" in k]
print(f"共 {len(swin_keys)} 个键:")
for k in swin_keys:
    print(f"  {k}  →  shape={model_dict[k].shape}")

# 4. 匹配分析
print("\n" + "=" * 60)
print("逐键匹配分析:")
print()

mapped = 0
mismatches = []
for k_model in sorted(swin_keys):
    # strip backbone. prefix (实际没有)
    k_pure = k_model

    # try layers2 (dim=96, 适合 VSNet dim=96)
    candidate = k_pure.replace("swintransformer", "swinViT.layers2")
    if candidate in state_dict:
        if model_dict[k_model].shape == state_dict[candidate].shape:
            mapped += 1
            print(f"  [OK] {k_model}")
            print(f"       ← {candidate}  shape={state_dict[candidate].shape}")
            continue
        else:
            mismatches.append((k_model, candidate, model_dict[k_model].shape, state_dict[candidate].shape))

    # try layers3
    candidate = k_pure.replace("swintransformer", "swinViT.layers3")
    if candidate in state_dict:
        if model_dict[k_model].shape == state_dict[candidate].shape:
            mapped += 1
            print(f"  [OK] {k_model}")
            print(f"       ← {candidate}  shape={state_dict[candidate].shape}")
            continue
        else:
            mismatches.append((k_model, candidate, model_dict[k_model].shape, state_dict[candidate].shape))

    print(f"  [X] {k_model}  (无匹配)")

print(f"\n映射成功: {mapped}/{len(swin_keys)}")

if mismatches:
    print(f"\n形状不匹配 ({len(mismatches)} 个):")
    for mk, ck, ms, cs in mismatches:
        print(f"  VSNet:     {mk}  {ms}")
        print(f"  Pretrain:  {ck}  {cs}")
        print()

# 5. 额外: 检查 layers1/layers2/layers3/layers4 dim 差异
print("\n" + "=" * 60)
print("各 layers 的 dim 信息:")
for layer in ["layers1", "layers2", "layers3", "layers4"]:
    keys = [k for k in swinvit_keys if layer in k]
    norm_keys = [k for k in keys if "norm" in k and "weight" in k]
    if norm_keys:
        print(f"  {layer}: {len(keys)} keys,  sample norm shape={state_dict[norm_keys[0]].shape}")