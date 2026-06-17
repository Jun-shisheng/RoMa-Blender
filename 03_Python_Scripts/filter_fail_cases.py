# -*- coding: utf-8 -*-
# 失效匹配案例筛选脚本（适配考场项目）
import os
import csv
import json
from pathlib import Path

# 配置参数（与集成方案一致）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "02_Data", "match_results", "car_match_report_precise.csv")
FAIL_CASE_DIR = os.path.join(ROOT_DIR, "02_Data", "fail_cases")
CONFIDENCE_THRESHOLD = 0.3  # 低置信阈值
FAIL_RATIO_THRESHOLD = 0.3  # 失效占比>30%标记为挂科

# 创建失效案例目录
Path(FAIL_CASE_DIR).mkdir(parents=True, exist_ok=True)


def calculate_fail_ratio(total_matches, low_conf_matches):
    """计算失效匹配占比（低置信点/总匹配点）"""
    if total_matches == 0:
        return 1.0  # 无匹配点视为完全失效
    return low_conf_matches / total_matches


def filter_fail_cases():
    fail_cases = []
    normal_cases = []

    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 解析数据
            try:
                match_id = row["匹配序号"]
                img1_name = row["第一张图（序号）"]
                img2_name = row["第二张图（序号）"]
                total_matches = int(row["总匹配点"])
                low_conf_matches = int(row["低置信度匹配点"])
                low_conf_rate = float(row["低置信率(%)"])
            except Exception as e:
                print(f"⚠️  跳过无效行：{row}，错误：{e}")
                continue

            # 计算失效占比并判断是否挂科
            fail_ratio = calculate_fail_ratio(total_matches, low_conf_matches)
            is_fail = fail_ratio > FAIL_RATIO_THRESHOLD
            case_data = {
                "match_id": match_id,
                "img1_name": img1_name,
                "img2_name": img2_name,
                "total_matches": total_matches,
                "low_conf_matches": low_conf_matches,
                "low_conf_rate": low_conf_rate,
                "fail_ratio": round(fail_ratio * 100, 2),
                "is_fail": is_fail
            }

            if is_fail:
                fail_cases.append(case_data)
            else:
                normal_cases.append(case_data)

    # 保存失效案例清单（JSON+CSV）
    # 1. JSON格式（便于Blender导入可视化）
    fail_json_path = os.path.join(FAIL_CASE_DIR, "fail_cases.json")
    with open(fail_json_path, "w", encoding="utf-8") as f:
        json.dump(fail_cases, f, ensure_ascii=False, indent=2)

    # 2. CSV格式（便于统计分析）
    fail_csv_path = os.path.join(FAIL_CASE_DIR, "fail_cases.csv")
    with open(fail_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "匹配序号", "图片1", "图片2", "总匹配点", "低置信点",
            "低置信率(%)", "失效占比(%)", "是否挂科"
        ])
        for case in fail_cases:
            writer.writerow([
                case["match_id"], case["img1_name"], case["img2_name"],
                case["total_matches"], case["low_conf_matches"],
                case["low_conf_rate"], case["fail_ratio"],
                "是" if case["is_fail"] else "否"
            ])

    # 输出统计结果
    print("📊 失效匹配筛选完成")
    print(f"总匹配对数：{len(fail_cases) + len(normal_cases)}")
    print(f"挂科案例数：{len(fail_cases)}（失效占比>30%）")
    print(f"正常案例数：{len(normal_cases)}")
    print(f"挂科率：{len(fail_cases) / (len(fail_cases) + len(normal_cases)) * 100:.2f}%")
    print(f"\n📁 失效案例文件保存至：{FAIL_CASE_DIR}")


if __name__ == "__main__":
    print("🚀 启动失效匹配案例筛选脚本")
    filter_fail_cases()