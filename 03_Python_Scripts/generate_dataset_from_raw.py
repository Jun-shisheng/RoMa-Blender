# 03_Python_Scripts/generate_dataset_from_raw.py
import os
import csv
import shutil
from pathlib import Path

# 配置路径（与项目目录完全匹配）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_METADATA_PATH = os.path.join(ROOT_DIR, "02_Data", "match_results", "car_match_report_precise.csv")
RENDERED_IMAGES_DIR = os.path.join(ROOT_DIR, "02_Data", "rendered_images", "car_scene")
GENERATED_DATA_ROOT = os.path.join(ROOT_DIR, "04_TrainingData", "roma_self_supervised_dataset")
IMAGE_PAIRS_DIR = os.path.join(GENERATED_DATA_ROOT, "image_pairs")
MASKS_DIR = os.path.join(GENERATED_DATA_ROOT, "masks")
GENERATED_METADATA_PATH = os.path.join(GENERATED_DATA_ROOT, "metadata.csv")

# 创建目录
Path(IMAGE_PAIRS_DIR).mkdir(parents=True, exist_ok=True)
Path(MASKS_DIR).mkdir(parents=True, exist_ok=True)

def generate_training_metadata():
    with open(RAW_METADATA_PATH, "r", encoding="utf-8") as raw_f, \
         open(GENERATED_METADATA_PATH, "w", encoding="utf-8", newline="") as gen_f:
        raw_reader = csv.DictReader(raw_f)
        gen_writer = csv.DictWriter(gen_f, fieldnames=[
            "case_id", "img1_path", "img2_path", "mask_path", "is_low_conf"
        ])
        gen_writer.writeheader()

        for row in raw_reader:
            case_id = row["匹配序号"]
            low_conf_rate = float(row["低置信率(%)"]) / 100
            is_low_conf = low_conf_rate > 0.03  # 3%阈值

            # 原始图像路径（直接使用序号作为文件名，无需额外加 .png）
            img1_name = row["第一张图（序号）"]  # 例如："0001.png"（从CSV中读取的已含后缀）
            img2_name = row["第二张图（序号）"]  # 例如："0020.png"
            img1_raw = os.path.join(RENDERED_IMAGES_DIR, img1_name)
            img2_raw = os.path.join(RENDERED_IMAGES_DIR, img2_name)

            # 检查图像是否存在
            if not os.path.exists(img1_raw) or not os.path.exists(img2_raw):
                print(f"⚠️  跳过案例{case_id}：图像{img1_name}/{img2_name}不存在（实际路径：{img1_raw}）")
                continue

            # 检查图像是否存在
            if not os.path.exists(img1_raw) or not os.path.exists(img2_raw):
                print(f"⚠️  跳过案例{case_id}：图像{img1_name}/{img2_name}不存在")
                continue

            # 复制图像到训练目录
            img1_gen = os.path.join(IMAGE_PAIRS_DIR, f"case_{case_id}_img1.png")
            img2_gen = os.path.join(IMAGE_PAIRS_DIR, f"case_{case_id}_img2.png")
            shutil.copyfile(img1_raw, img1_gen)
            shutil.copyfile(img2_raw, img2_gen)

            # 掩码路径（低置信案例需提前生成掩码，命名为“case_xx_low_conf_mask.png”）
            mask_gen = os.path.join(MASKS_DIR, f"case_{case_id}_low_conf_mask.png") if is_low_conf else ""
            if is_low_conf and not os.path.exists(mask_gen):
                print(f"⚠️  案例{case_id}（低置信）：掩码{mask_gen}不存在，请先运行可视化脚本生成")
                continue

            # 写入元数据
            gen_writer.writerow({
                "case_id": case_id,
                "img1_path": img1_gen,
                "img2_path": img2_gen,
                "mask_path": mask_gen,
                "is_low_conf": str(is_low_conf)
            })

    print(f"✅ 数据集生成完成：{GENERATED_DATA_ROOT}")
    print(f"  - 元数据：{GENERATED_METADATA_PATH}（{len(list(csv.DictReader(open(GENERATED_METADATA_PATH))))} 条记录）")
    print(f"  - 图像对：{len(os.listdir(IMAGE_PAIRS_DIR))//2} 对")
    print(f"  - 掩码：{len(os.listdir(MASKS_DIR))} 个（低置信案例）")

if __name__ == "__main__":
    generate_training_metadata()