# 03_Python_Scripts/evaluate_roma.py
import os
import csv
import torch
import cv2
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn  # 修复未解析的nn引用
from torchvision import models

# ============ 全局配置（与训练脚本完全对齐） ============
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNED_MODEL_PATH = os.path.join(ROOT_DIR, "04_TrainingData", "finetuned_roma", "best_finetuned_roma.pth")
RAW_METADATA_PATH = os.path.join(ROOT_DIR, "02_Data", "match_results", "car_match_report_precise.csv")
OUTPUT_REPORT_PATH = os.path.join(ROOT_DIR, "04_TrainingData", "evaluation_report.csv")
RENDERED_IMAGES_DIR = os.path.join(ROOT_DIR, "02_Data", "rendered_images", "car_scene")

# ============ 核心修复：添加正确的romatch模块路径 ============
import sys
sys.path.append(ROOT_DIR)
try:
    from romatch.models.model_zoo.roma_models import RegressionMatcher
    from romatch.models.encoders import CNNandDinov2
    print("✅ romatch模块引用成功！")
except ImportError as e:
    print(f"❌ 未找到romatch模块，请确保路径正确：")
    print(f"  - 当前添加的路径：{project_root}")
    print(f"  - 请确认该路径下存在romatch文件夹（训练时使用的模块）")
    raise

# ============ 核心修复：重构模型加载逻辑（与训练脚本一致） ============
def init_evaluator_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    FEATURE_DIM = 64

    class FakeMultiScaleEncoder(nn.Module):
        def __init__(self, output_dim=FEATURE_DIM):
            super().__init__()
            self.cnn = models.vgg19_bn(weights=None)
            self.feature_extractor = nn.Sequential(
                *list(self.cnn.features.children())[:23],
                nn.Conv2d(256, output_dim, 1, 1, 0)
            )

        def forward(self, x, **kwargs):
            feat = self.feature_extractor(x)
            return {"single_scale": feat}

    class MatchDecoder(nn.Module):
        def __init__(self, input_dim=FEATURE_DIM, output_dim=2):
            super().__init__()
            self.decoder = nn.Sequential(
                nn.Conv2d(input_dim, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, output_dim, 1, 1, 0)
            )

        def forward(self, x, *args, **kwargs):
            if isinstance(x, dict):
                x = next(iter(x.values()))
            return self.decoder(x)

    encoder = FakeMultiScaleEncoder().to(DEVICE)
    decoder = MatchDecoder().to(DEVICE)
    model = RegressionMatcher(
        encoder=encoder,
        decoder=decoder,
        upsample_preds=True
    ).to(DEVICE)

    checkpoint = torch.load(FINETUNED_MODEL_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ 微调模型加载成功（设备：{DEVICE}）")
    return model

def evaluate_roma():
    model = init_evaluator_model()
    DEVICE = next(model.parameters()).device

    if not os.path.exists(RAW_METADATA_PATH):
        raise FileNotFoundError(f"原始匹配报告不存在：{RAW_METADATA_PATH}")
    with open(RAW_METADATA_PATH, "r", encoding="utf-8") as f:
        raw_reader = csv.DictReader(f)
        cases = list(raw_reader)
    print(f"✅ 加载{len(cases)}个评估案例")

    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH), exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "original_low_conf_rate", "finetuned_low_conf_rate", "improvement"
        ])
        writer.writeheader()

        for case in tqdm(cases, desc="评估案例", colour='green'):
            case_id = case["匹配序号"]
            original_low_conf = case["低置信率(%)"]
            original_low_conf = float(original_low_conf) / 100 if original_low_conf and original_low_conf != "NA" else 0.0

            img1_name = f"{case['第一张图（序号）']}"
            img2_name = f"{case['第二张图（序号）']}"
            img1_path = os.path.join(RENDERED_IMAGES_DIR, img1_name)
            img2_path = os.path.join(RENDERED_IMAGES_DIR, img2_name)

            if not os.path.exists(img1_path) or not os.path.exists(img2_path):
                print(f"⚠️  跳过案例{case_id}：图像{img1_name}/{img2_name}不存在")
                continue

            def preprocess_img(img_path):
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                return img_tensor.to(DEVICE)

            img1_tensor = preprocess_img(img1_path)
            img2_tensor = preprocess_img(img2_path)

            with torch.no_grad():
                outputs = model({"im_A": img1_tensor, "im_B": img2_tensor})

            # 修复：添加detach()以解决梯度依赖问题
            if isinstance(outputs, dict):
                certainty = outputs.get("certainty", None)
                if certainty is not None:
                    certainty = certainty.squeeze().detach().cpu().numpy()
                else:
                    featA = next(iter(outputs.values())).detach()
                    featB = model({"im_A": img2_tensor, "im_B": img1_tensor})
                    featB = next(iter(featB.values())).detach()
                    certainty = torch.nn.functional.cosine_similarity(featA, featB, dim=1).detach().cpu().numpy()
            else:
                featA = outputs.detach()
                featB = model({"im_A": img2_tensor, "im_B": img1_tensor}).detach()
                certainty = torch.nn.functional.cosine_similarity(featA, featB, dim=1).detach().cpu().numpy()

            finetuned_low_conf = np.mean(certainty < 0.3) if certainty.size > 0 else 0.0
            improvement = original_low_conf - finetuned_low_conf

            writer.writerow({
                "case_id": case_id,
                "original_low_conf_rate": f"{original_low_conf:.4f}",
                "finetuned_low_conf_rate": f"{finetuned_low_conf:.4f}",
                "improvement": f"{improvement:.4f}"
            })

    print(f"\n🎉 评估完成！")
    print(f"📄 评估报告路径：{OUTPUT_REPORT_PATH}")
    print("📌 关键说明：")
    print("  - improvement > 0：微调后低置信率下降（模型优化有效）")
    print("  - improvement = 0：无变化")
    print("  - improvement < 0：微调后效果变差（需检查数据/模型）")

if __name__ == "__main__":
    evaluate_roma()