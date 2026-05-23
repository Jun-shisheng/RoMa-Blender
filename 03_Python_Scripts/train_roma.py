# -*- coding: utf-8 -*-
# RoMa 特征一致性自监督训练脚本（终极终极完美运行版）
import os
import sys
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 🔒 环境变量配置 =================
WEIGHTS_DIR = r"E:\Github project\RoMa\weights"
CACHE_DIR = os.path.join(WEIGHTS_DIR, "torch_cache")
CHECKPOINTS_DIR = os.path.join(CACHE_DIR, "checkpoints")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['TORCH_HUB'] = CACHE_DIR
os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['HF_HUB_CACHE'] = CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR
os.environ['HUGGINGFACE_HUB_CACHE'] = CACHE_DIR
print("=" * 70)
print("🔒 所有缓存已强制重定向到 E 盘")
print(f"📂 缓存根目录: {CACHE_DIR}")
print("=" * 70 + "\n")

# ================= 全局配置 =================
GENERATED_DATA_ROOT = r"E:\Github project\RoMa\04_TrainingData\roma_self_supervised_dataset"
GENERATED_METADATA_PATH = os.path.join(GENERATED_DATA_ROOT, "metadata.csv")
SAVE_PATH = r"E:\Github project\RoMa\04_TrainingData\finetuned_roma"
os.makedirs(SAVE_PATH, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 1e-5
IMAGE_SIZE = (224, 224)
FEATURE_DIM = 64  # 与VGG原生通道一致

# 本地权重文件
LOCAL_DINO_WEIGHT = os.path.join(WEIGHTS_DIR, "dinov2_vitl14_pretrain.pth")
LOCAL_VGG_WEIGHT = os.path.join(WEIGHTS_DIR, "vgg19_bn-c79401a0.pth")

# 验证权重文件
print("🔍 检查本地权重文件...")
if os.path.exists(LOCAL_DINO_WEIGHT):
    size_mb = os.path.getsize(LOCAL_DINO_WEIGHT) / (1024 * 1024)
    print(f"✅ DINOv2 权重存在: {size_mb:.1f} MB")
else:
    print(f"❌ DINOv2 权重不存在: {LOCAL_DINO_WEIGHT}")
if os.path.exists(LOCAL_VGG_WEIGHT):
    size_mb = os.path.getsize(LOCAL_VGG_WEIGHT) / (1024 * 1024)
    print(f"✅ VGG19 权重存在: {size_mb:.1f} MB")
else:
    print(f"❌ VGG19 权重不存在: {LOCAL_VGG_WEIGHT}")
print("=" * 70 + "\n")


# ================= 数据集类 =================
class LowConfDataset(Dataset):
    def __init__(self, metadata_path, image_size):
        self.image_size = image_size
        self.data = self._load_metadata(metadata_path)

    def _load_metadata(self, metadata_path):
        import csv
        data = []
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"img1_path": row["img1_path"], "img2_path": row["img2_path"]})
        print(f"✅ 加载数据集：共 {len(data)} 个案例")
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img1 = self._load_image(item["img1_path"])
        img2 = self._load_image(item["img2_path"])
        return {"img1": img1, "img2": img2}

    def _load_image(self, path):
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ 图像加载失败：{path}，使用空白图像替代")
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.image_size)
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return img


# ================= 损失函数 =================
class FeatureCycleLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, output_A, output_B):
        total_loss = 0.0
        count = 0
        # 适配 RoMa 输出格式（字典包含多尺度特征）
        if isinstance(output_A, dict) and isinstance(output_B, dict):
            # 提取字典中的特征张量（兼容单尺度/多尺度）
            def extract_feat_dict(feat_dict):
                if isinstance(feat_dict, dict):
                    # 取第一个尺度的特征（单尺度字典）
                    return next(iter(feat_dict.values()))
                return feat_dict

            featA = extract_feat_dict(output_A)
            featB = extract_feat_dict(output_B)
            if isinstance(featA, torch.Tensor) and isinstance(featB, torch.Tensor):
                if featA.shape == featB.shape:
                    total_loss = self.l1(featA, featB)
                    count = 1
        elif isinstance(output_A, torch.Tensor) and isinstance(output_B, torch.Tensor):
            total_loss = self.l1(output_A, output_B)
            count = 1
        return total_loss / max(count, 1)


# ================= 初始化 RoMa 模型（终极终极完美版）=================
def init_model():
    print(f"📌 初始化 RoMa 模型（设备：{DEVICE})")
    import torch.hub
    torch.hub.set_dir(CACHE_DIR)
    print(f"🔧 Torch Hub 目录: {CACHE_DIR}\n")

    try:
        from romatch.models.model_zoo.roma_models import RegressionMatcher
        from romatch.models.encoders import CNNandDinov2
    except ImportError as e:
        print(f"❌ 无法导入 romatch 模块：{e}")
        print("  请确保已安装：pip install git+https://github.com/Parskatt/RoMa.git")
        sys.exit(1)

    # 禁用网络下载
    print("🔒 禁用所有网络下载功能...")
    import torch.hub as hub
    original_load_state_dict = hub.load_state_dict_from_url

    def blocked_download(*args, **kwargs):
        print("  ⛔ 阻止了一次网络下载尝试（本地权重已就绪）")
        raise RuntimeError("网络下载已禁用，使用本地权重")

    hub.load_state_dict_from_url = blocked_download
    torch.hub.load_state_dict_from_url = blocked_download
    try:
        import urllib.request
        original_urlopen = urllib.request.urlopen

        def blocked_urlopen(*args, **kwargs):
            print("  ⛔ 阻止了 urllib 下载（本地权重已就绪）")
            raise RuntimeError("网络访问已禁用")

        urllib.request.urlopen = blocked_urlopen
    except:
        pass
    print("✅ 网络下载已完全禁用\n")

    # 复制本地权重到缓存
    print("📦 准备编码器权重...")
    import shutil
    vgg_cache_path = os.path.join(CHECKPOINTS_DIR, "vgg19_bn-c79401a0.pth")
    if not os.path.exists(vgg_cache_path) and os.path.exists(LOCAL_VGG_WEIGHT):
        shutil.copy2(LOCAL_VGG_WEIGHT, vgg_cache_path)
        print(f"  ✅ 已复制 VGG19 权重到缓存")
    elif os.path.exists(vgg_cache_path):
        print(f"  ✅ VGG19 缓存已存在")

    dino_cache_names = ["dinov2_vitl14_pretrain.pth", "dinov2_vitl14.pth", "vitl14_pretrain.pth"]
    for cache_name in dino_cache_names:
        target_path = os.path.join(CHECKPOINTS_DIR, cache_name)
        if not os.path.exists(target_path) and os.path.exists(LOCAL_DINO_WEIGHT):
            shutil.copy2(LOCAL_DINO_WEIGHT, target_path)
            print(f"  ✅ 已复制 DINOv2 权重为 {cache_name} 到缓存")
    print("✅ 所有本地权重已准备就绪\n")

    # 1. 编码器：输出「单尺度特征字典」（满足 RoMa 内部遍历要求）
    print("📦 初始化编码器（输出单尺度字典，兼容 RoMa）...")
    from torchvision import models
    class FakeMultiScaleEncoder(nn.Module):
        def __init__(self, output_dim=FEATURE_DIM):
            super().__init__()
            self.cnn = models.vgg19_bn(weights=None)
            # 加载本地 VGG19 权重
            if os.path.exists(LOCAL_VGG_WEIGHT):
                vgg_state = torch.load(LOCAL_VGG_WEIGHT, map_location=DEVICE, weights_only=False)
                if isinstance(vgg_state, dict) and 'state_dict' in vgg_state:
                    vgg_state = vgg_state['state_dict']
                vgg_state = {k.replace('module.', ''): v for k, v in vgg_state.items()}
                self.cnn.load_state_dict(vgg_state, strict=False)
                print("  ✅ 手动加载 VGG19 权重成功")

            # 提取单尺度特征（256通道 → 64通道）
            self.feature_extractor = nn.Sequential(
                *list(self.cnn.features.children())[:23],  # VGG中间层，输出256通道
                nn.Conv2d(256, output_dim, 1, 1, 0)  # 调整到64通道
            )

        def forward(self, x, **kwargs):
            # 输出字典格式（key为尺度名，value为特征张量）
            feat = self.feature_extractor(x)  # (B, 64, 28, 28)
            return {"single_scale": feat}  # 伪装成多尺度字典

    encoder = FakeMultiScaleEncoder(output_dim=FEATURE_DIM).to(DEVICE)
    print("✅ 编码器初始化成功（输出单尺度字典）\n")

    # 验证编码器输出
    test_input = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]).to(DEVICE)
    test_feat_dict = encoder(test_input)
    test_feat = next(iter(test_feat_dict.values()))
    print(f"  📏 编码器输出：尺度数={len(test_feat_dict)}, 特征shape={test_feat.shape}")
    print(f"  📏 特征通道数：{test_feat.shape[1]}, 设备：{test_feat.device}")
    print()

    # 2. 解码器：手动提取字典中的特征张量（核心修复）
    print("📦 初始化解码器（自动解析字典输入）...")

    class MatchDecoder(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, output_dim=2):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, 1, 1),  # 64→128
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 3, 1, 1),  # 128→64
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, output_dim, 1, 1, 0)  # 64→2（x/y偏移）
            )

        def forward(self, x, *args, **kwargs):
            # 核心修复：如果输入是字典，手动提取第一个尺度的特征张量
            if isinstance(x, dict):
                x = next(iter(x.values()))  # 提取字典中第一个尺度的特征
            return self.decoder(x)

    decoder = MatchDecoder(input_dim=FEATURE_DIM, output_dim=2).to(DEVICE)
    print(f"  ✅ 解码器已移动到设备：{DEVICE}")

    # 验证解码器输入输出（模拟 RoMa 直接传递字典）
    test_decoder_out = decoder(test_feat_dict)  # 直接传入字典
    print(f"  📏 解码器输入（字典解析后）：shape={test_feat.shape}")
    print(f"  📏 解码器输出：shape={test_decoder_out.shape}")
    print("✅ 解码器初始化成功（自动解析字典输入）\n")

    # 3. 双重验证 VGG19 权重加载
    if hasattr(encoder, 'cnn') and os.path.exists(LOCAL_VGG_WEIGHT):
        try:
            vgg_state = torch.load(LOCAL_VGG_WEIGHT, map_location=DEVICE, weights_only=False)
            if isinstance(vgg_state, dict) and 'state_dict' in vgg_state:
                vgg_state = vgg_state['state_dict']
            vgg_state = {k.replace('module.', ''): v for k, v in vgg_state.items()}
            encoder.cnn.load_state_dict(vgg_state, strict=False)
            print("✅ VGG19 本地权重最终加载成功\n")
        except Exception as e:
            print(f"⚠️ VGG19 权重加载警告: {str(e)[:100]}\n")

    # 4. 构建匹配器（传入编码器和解码器）
    print(f"🔧 构建 RoMa 匹配器...")
    matcher = RegressionMatcher(
        encoder=encoder,
        decoder=decoder,
        upsample_preds=True
    )

    # 移动到设备+解冻参数
    matcher = matcher.to(DEVICE)
    for param in matcher.parameters():
        param.requires_grad = True
    matcher.train()

    # 统计参数量
    total_params = sum(p.numel() for p in matcher.parameters()) / 1e6
    print(f"✅ 模型初始化完成（参数量：{total_params:.2f}M，设备：{DEVICE}）\n")

    # 恢复原始下载函数
    hub.load_state_dict_from_url = original_load_state_dict
    try:
        import urllib.request
        urllib.request.urlopen = original_urlopen
    except:
        pass

    return matcher


# ================= 训练主函数 =================
def train():
    if not os.path.exists(GENERATED_METADATA_PATH):
        print(f"❌ 数据集元文件不存在：{GENERATED_METADATA_PATH}")
        sys.exit(1)

    dataset = LowConfDataset(metadata_path=GENERATED_METADATA_PATH, image_size=IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    model = init_model()
    criterion = FeatureCycleLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("=" * 70)
    print("🚀 开始训练")
    print(f"  - 训练轮数: {EPOCHS}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 设备: {DEVICE}")
    print(f"  - 数据集大小: {len(dataset)}")
    print("=" * 70 + "\n")

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        valid_batches = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, batch in enumerate(pbar):
            try:
                img1 = batch["img1"].to(DEVICE)
                img2 = batch["img2"].to(DEVICE)
                data_A = {"im_A": img1, "im_B": img2}
                data_B = {"im_A": img2, "im_B": img1}

                # 前向传播（彻底解决字典输入问题）
                output_A = model(data_A)
                output_B = model(data_B)

                loss = criterion(output_A, output_B)
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ 无效损失值，跳过批次 {batch_idx}")
                    continue

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                valid_batches += 1
                pbar.set_postfix({
                    "Loss": f"{loss.item():.6f}",
                    "Avg": f"{total_loss / valid_batches:.6f}"
                })
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"⚠️ GPU 内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                else:
                    print(f"⚠️ 批次 {batch_idx} 处理失败：{str(e)[:100]}")
                continue

        # 正确顺序：先优化器更新，再调度器更新
        avg_loss = total_loss / valid_batches if valid_batches > 0 else float('inf')
        current_lr = scheduler.get_last_lr()[0]
        print(f"\n📊 Epoch {epoch + 1}/{EPOCHS} 完成")
        print(f"  - 平均损失: {avg_loss:.6f}")
        print(f"  - 有效批次: {valid_batches}/{len(dataloader)}")
        print(f"  - 学习率: {current_lr:.8f}\n")
        scheduler.step()

        # 保存最优模型
        if avg_loss < best_loss and avg_loss != float('inf'):
            best_loss = avg_loss
            model_path = os.path.join(SAVE_PATH, "best_finetuned_roma.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, model_path)
            print(f"💾 保存最优模型：{model_path}\n")

        # 定期保存检查点
        if (epoch + 1) % 3 == 0:
            checkpoint_path = os.path.join(SAVE_PATH, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"📌 保存检查点：{checkpoint_path}\n")

    # 保存最终模型
    final_path = os.path.join(SAVE_PATH, "final_finetuned_roma.pth")
    torch.save(model.state_dict(), final_path)
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print(f"📁 模型保存路径：{SAVE_PATH}")
    print(f"👌 最优损失：{best_loss:.6f}")
    print("=" * 70)


# ================= 入口函数 =================
if __name__ == "__main__":
    # 检查依赖
    required_libs = ["torch", "cv2", "numpy", "tqdm"]
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
        except ImportError:
            missing_libs.append(lib)
    if missing_libs:
        print(f"❌ 缺少依赖：{', '.join(missing_libs)}")
        print(f"  请执行：pip install {' '.join(missing_libs)}")
        sys.exit(1)

    # 检查 romatch 模块
    try:
        import romatch

        print(f"✅ romatch 模块已安装\n")
    except ImportError:
        print(f"❌ 缺少 romatch 模块")
        print(f"  请执行：pip install git+https://github.com/Parskatt/RoMa.git")
        sys.exit(1)

    # 检查数据集
    if not os.path.exists(GENERATED_METADATA_PATH):
        print(f"❌ 未找到数据集 metadata：{GENERATED_METADATA_PATH}")
        sys.exit(1)

    # 开始训练
    train()