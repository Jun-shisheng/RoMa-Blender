# -*- coding: utf-8 -*-
# 项目CSV报告查看工具（最终稳定版：显示全部+提示同步）
import os
import csv
from typing import List, Dict

# ============ 路径配置（与项目保持一致） ============
ROOT_DIR = r"E:\Github project\RoMa"
REPORT_PATH = os.path.join(
    ROOT_DIR, "02_Data", "match_results", "car_match_report_precise.csv"
)


# ============ 核心功能函数 ============
def load_csv_data() -> List[Dict]:
    """加载CSV报告数据，返回字典列表（每行为一个字典）"""
    if not os.path.exists(REPORT_PATH):
        raise FileNotFoundError(f"未找到CSV报告文件：{REPORT_PATH}")

    data = []
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)  # 按列名读取，无需关心列顺序
        for row in reader:
            # 转换数值类型（兼容CSV实际列名）
            try:
                row["总匹配点"] = int(row["总匹配点"]) if row["总匹配点"] != "异常" else None
                row["低置信度匹配点"] = int(row["低置信度匹配点"]) if row["低置信度匹配点"] != "异常" else None
                row["低置信率(%)"] = float(row["低置信率(%)"]) if row["低置信率(%)"] != "异常" else None
            except Exception as e:
                print(f"⚠️  解析行数据失败（跳过）：{row}，错误：{e}")
                continue
            data.append(row)
    return data


def print_overall_stats(data: List[Dict]):
    """打印整体统计信息（复刻项目汇总逻辑）"""
    total_pairs = len(data)
    valid_pairs = [row for row in data if row["低置信率(%)"] is not None]
    fail_pairs = [row for row in valid_pairs if row["低置信率(%)"] > 50]

    avg_low_conf = (sum(row["低置信率(%)"] for row in valid_pairs) / len(valid_pairs)) if valid_pairs else 0.0

    print("=" * 50)
    print("📊  CSV报告整体统计")
    print("=" * 50)
    print(f"总匹配对数：{total_pairs}")
    print(f"有效匹配对数（无异常）：{len(valid_pairs)}")
    print(f"低置信率过高对数（>50%）：{len(fail_pairs)}")
    print(f"平均低置信率：{avg_low_conf:.2f}%")
    print(f"最高低置信率：{max(row['低置信率(%)'] for row in valid_pairs):.2f}%" if valid_pairs else "无")
    print(f"最低低置信率：{min(row['低置信率(%)'] for row in valid_pairs):.2f}%" if valid_pairs else "无")
    print("=" * 50 + "\n")


def filter_matches(data: List[Dict], min_confidence: float = None, max_confidence: float = None,
                   show_all: bool = False):
    """筛选匹配对（支持按低置信率范围过滤，可选择显示全部）"""
    filtered = data
    if min_confidence is not None:
        filtered = [row for row in filtered if row["低置信率(%)"] is not None and row["低置信率(%)"] >= min_confidence]
    if max_confidence is not None:
        filtered = [row for row in filtered if row["低置信率(%)"] is not None and row["低置信率(%)"] <= max_confidence]

    print(f"🔍  筛选结果（低置信率范围：{min_confidence or '无'}~{max_confidence or '无'}%）")
    if not filtered:
        print("  无符合条件的匹配对")
        return

    # 按低置信率升序排序（最好的结果在前）
    filtered_sorted = sorted(filtered, key=lambda x: x["低置信率(%)"])
    total = len(filtered_sorted)

    # 选择显示全部或前10条
    if show_all:
        for i, row in enumerate(filtered_sorted, 1):
            print(
                f"  第{i}对：{row['第一张图（序号）']} ↔ {row['第二张图（序号）']} | 低置信率：{row['低置信率(%)']:.2f}% | 状态：{row['匹配状态']}")
    else:
        # 显示前10条
        for i, row in enumerate(filtered_sorted[:10], 1):
            print(
                f"  第{i}对：{row['第一张图（序号）']} ↔ {row['第二张图（序号）']} | 低置信率：{row['低置信率(%)']:.2f}% | 状态：{row['匹配状态']}")
        if total > 10:
            print(f"  ... 共{total}条符合条件的匹配对（已显示前10条）")
            print(f"  💡 提示：输入5可显示全部结果")

    print()


def search_match(data: List[Dict], img_name: str, show_all: bool = False):
    """搜索包含指定图片的匹配对（支持模糊搜索，可选择显示全部）"""
    matches = []
    for row in data:
        if img_name.lower() in row["第一张图（序号）"].lower() or img_name.lower() in row["第二张图（序号）"].lower():
            matches.append(row)

    print(f"🔎  搜索结果（包含图片：{img_name}）")
    if not matches:
        print("  无匹配结果")
        return

    # 按低置信率升序排序
    matches_sorted = sorted(matches, key=lambda x: x["低置信率(%)"])
    total = len(matches_sorted)

    # 选择显示全部或前10条
    if show_all:
        for i, row in enumerate(matches_sorted, 1):
            print(
                f"  第{i}对：{row['第一张图（序号）']} ↔ {row['第二张图（序号）']} | 低置信率：{row['低置信率(%)']:.2f}% | 状态：{row['匹配状态']}")
    else:
        for i, row in enumerate(matches_sorted[:10], 1):
            print(
                f"  第{i}对：{row['第一张图（序号）']} ↔ {row['第二张图（序号）']} | 低置信率：{row['低置信率(%)']:.2f}% | 状态：{row['匹配状态']}")
        if total > 10:
            print(f"  ... 共{total}条匹配结果（已显示前10条）")
            print(f"  💡 提示：输入5可显示全部结果")

    print()


# ============ 主交互逻辑 ============
def main():
    print("🚀  项目CSV报告查看工具（最终稳定版）")
    print(f"📁  正在加载报告：{REPORT_PATH}\n")

    try:
        # 加载数据
        data = load_csv_data()
        print(f"✅  成功加载 {len(data)} 条匹配记录\n")

        # 打印整体统计
        print_overall_stats(data)

        # 交互状态变量
        current_filtered = []  # 存储当前筛选/搜索的结果
        current_mode = "none"  # 记录当前模式：none/filter/search
        current_search_key = ""  # 记录当前搜索关键词
        current_min_conf = None
        current_max_conf = None

        while True:
            # 菜单提示（同步更新为1-5，避免误解）
            print("请选择操作：")
            print("1. 查看高质量匹配对（低置信率≤5%）")
            print("2. 查看所有匹配对（按低置信率排序）")
            print("3. 搜索指定图片的匹配记录")
            print("4. 退出工具")
            choice = input("\n输入操作序号（1-4）：").strip()  # 提示文字改为1-5

            if choice == "1":
                # 查看高质量匹配对（≤5%）
                current_mode = "filter"
                current_min_conf = None
                current_max_conf = 5.0
                filter_matches(data, min_confidence=current_min_conf, max_confidence=current_max_conf, show_all=False)
                current_filtered = [row for row in data if row["低置信率(%)"] is not None and row["低置信率(%)"] <= 5.0]
            elif choice == "2":
                # 查看所有匹配对
                current_mode = "filter"
                current_min_conf = None
                current_max_conf = None
                filter_matches(data, min_confidence=current_min_conf, max_confidence=current_max_conf, show_all=False)
                current_filtered = data
            elif choice == "3":
                # 搜索指定图片
                current_mode = "search"
                current_search_key = input("输入要搜索的图片名（如0001、0020）：").strip()
                search_match(data, current_search_key, show_all=False)
                current_filtered = [row for row in data if current_search_key.lower() in row[
                    "第一张图（序号）"].lower() or current_search_key.lower() in row["第二张图（序号）"].lower()]
            elif choice == "4":
                print("\n👋  退出工具，感谢使用！")
                break
            elif choice == "5":
                # 显示全部（需先执行1/2/3）
                if current_mode == "none":
                    print("❌  请先执行1（筛选高质量）、2（查看所有）或3（搜索）操作，再使用5显示全部\n")
                    continue

                print(f"🔍  显示全部{len(current_filtered)}条结果：")
                if current_mode == "filter":
                    filter_matches(data, min_confidence=current_min_conf, max_confidence=current_max_conf,
                                   show_all=True)
                elif current_mode == "search":
                    search_match(data, current_search_key, show_all=True)
            else:
                print("❌  输入无效，请重新选择1-5之间的序号\n")

    except KeyboardInterrupt:
        # 捕获手动中断，友好提示
        print("\n\n⚠️脚本已中断，如需继续请重新运行工具～")
    except Exception as e:
        print(f"\n❌工具运行失败：{str(e)}")


if __name__ == "__main__":
    main()