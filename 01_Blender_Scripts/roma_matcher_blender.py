# Blender内的RoMa匹配+分析脚本（精确间隔匹配版）
import torch
import os
import csv
import glob
from romatch import roma_outdoor

# 核心路径配置
ROOT_DIR = r"E:\Github project\RoMa"
RENDER_DIR = os.path.join(ROOT_DIR, "02_Data", "rendered_images", "car_scene")
RESULT_DIR = os.path.join(ROOT_DIR, "02_Data", "match_results")
FAIL_CASE_DIR = os.path.join(RESULT_DIR, "failure_cases")

# 创建结果文件夹
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FAIL_CASE_DIR, exist_ok=True)

# 关键配置（按需求精准设置）
CONFIDENCE_THRESHOLD = 0.3  # 低置信度阈值（<0.3视为低置信）
BASE_OFFSET = 19  # 起始图与匹配图的固定差值（1+19=20、5+19=25、10+19=30...）
MAX_MATCH_PAIRS = 50  # 最大匹配对数（防止超出图片范围）
START_FRAMES = []  # 存储所有起始图的索引（对应0001、0005、0010...）

# 加载并排序渲染图（0001.png ~ 0250.png）
image_paths = sorted(
    glob.glob(os.path.join(RENDER_DIR, "*.png")),
    key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))))  # 按文件名数字排序
)

# 过滤0001~0250的图片，并建立「图片序号→列表索引」的映射
filtered_images = []
img_num_to_idx = {}  # 键：图片序号（1→250），值：在filtered_images中的索引
for idx, img_path in enumerate(image_paths):
    img_name = os.path.basename(img_path)
    img_num = int(''.join(filter(str.isdigit, img_name)))  # 提取图片序号（如0001→1）
    if 1 <= img_num <= 250:
        filtered_images.append(img_path)
        img_num_to_idx[img_num] = idx

if not filtered_images:
    raise FileNotFoundError(f"未在 {RENDER_DIR} 下找到 0001.png ~ 0250.png 渲染图")

print(f"✅ 成功加载 {len(filtered_images)} 张渲染图（0001.png ~ 0250.png）")

# 生成起始图序号列表（1→5→10→15→20→25... 每次+5）
current_start_num = 1  # 第一个起始图序号：1（对应0001.png）
while current_start_num <= 250 - BASE_OFFSET:  # 确保匹配图不超过250（如250-19=231，起始图最大为231）
    START_FRAMES.append(current_start_num)
    current_start_num += 5  # 每次递增5（1→5→10→15...）

print(f"ℹ️  生成起始图序号：{START_FRAMES[:10]}...（共{len(START_FRAMES)}个起始图）")

# 初始化RoMa模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
roma_model = roma_outdoor(device=device)
print(f"✅ RoMa模型初始化完成，使用设备：{device}")

# 初始化匹配报告CSV（文件名区分精确匹配版）
report_csv = os.path.join(RESULT_DIR, "car_match_report_precise.csv")
with open(report_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "匹配序号", "第一张图（序号）", "第二张图（序号）",
        "总匹配点", "低置信度匹配点", "低置信率(%)", "匹配状态"
    ])

# 批量执行精确间隔匹配
match_count = 0
for start_num in START_FRAMES:
    if match_count >= MAX_MATCH_PAIRS:
        print(f"⚠️ 已达到最大匹配对数 {MAX_MATCH_PAIRS}，停止匹配")
        break

    # 计算匹配图序号（起始图序号 + 固定偏移19）
    match_num = start_num + BASE_OFFSET
    if match_num > 250:  # 确保匹配图不超出250
        print(f"⚠️ 匹配图序号 {match_num} 超出250，跳过该对")
        continue

    # 获取两张图在filtered_images中的索引
    img1_idx = img_num_to_idx[start_num]
    img2_idx = img_num_to_idx[match_num]
    img1_path = filtered_images[img1_idx]
    img2_path = filtered_images[img2_idx]
    img1_name = os.path.basename(img1_path)
    img2_name = os.path.basename(img2_path)
    match_count += 1

    print(f"\n--- 匹配第 {match_count} 对 ---")
    print(f"📷 图片对：{img1_name}（序号{start_num}） ↔ {img2_name}（序号{match_num}）")

    try:
        # 执行RoMa匹配
        warp, certainty = roma_model.match(img1_path, img2_path, device=device)
        matches, certainty = roma_model.sample(warp, certainty)

        # 计算匹配指标
        total_matches = len(matches)
        low_conf_matches = len(matches[certainty < CONFIDENCE_THRESHOLD]) if total_matches > 0 else 0
        low_conf_rate = (low_conf_matches / total_matches * 100) if total_matches > 0 else 0.0
        match_status = "正常" if low_conf_rate <= 50 else "低置信率过高"

        # 打印详细结果
        print(f"  总匹配点：{total_matches}")
        print(f"  低置信度匹配点：{low_conf_matches}")
        print(f"  低置信率：{low_conf_rate:.2f}%")
        print(f"  匹配状态：{match_status}")

        # 写入CSV报告
        with open(report_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                match_count, img1_name, img2_name,
                total_matches, low_conf_matches, round(low_conf_rate, 2), match_status
            ])

        # 保存低置信率过高的失败案例
        if low_conf_rate > 50:
            fail_pair_dir = os.path.join(FAIL_CASE_DIR, f"pair_{match_count}_{start_num}_vs_{match_num}")
            os.makedirs(fail_pair_dir, exist_ok=True)
            print(f"  ⚠️ 已保存失败案例到：{fail_pair_dir}")

    except Exception as e:
        print(f"❌ 第 {match_count} 对匹配失败：{str(e)}")
        # 写入失败记录到CSV
        with open(report_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                match_count, img1_name, img2_name,
                "异常", "异常", "异常", f"匹配失败：{str(e)}"
            ])

# 释放资源 + 生成整体统计汇总
del roma_model
torch.cuda.empty_cache()

# 读取CSV计算整体匹配效果
total_pairs = 0
total_low_conf_rate = 0.0
fail_pairs = 0
with open(report_csv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        total_pairs += 1
        if row["低置信率(%)"] != "异常":
            total_low_conf_rate += float(row["低置信率(%)"])
            if float(row["低置信率(%)"]) > 50:
                fail_pairs += 1

avg_low_conf_rate = (total_low_conf_rate / total_pairs) if total_pairs > 0 else 0.0
print(f"\n📊 匹配分析汇总")
print(f"总匹配对数：{total_pairs}")
print(f"平均低置信率：{avg_low_conf_rate:.2f}%")
print(f"低置信率过高的对数：{fail_pairs}")
print(f"\n✅ 所有匹配完成！")
print(f"📁 详细报告路径：{report_csv}")
print(f"📁 失败案例路径：{FAIL_CASE_DIR}")