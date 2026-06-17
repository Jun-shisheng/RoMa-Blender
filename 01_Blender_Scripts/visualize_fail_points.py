# -*- coding: utf-8 -*-
# Blender低置信区域可视化脚本（小球优化版：兼容旧版本+小巧半透明）
import bpy
import os
import json
import csv
import cv2
import numpy as np
from mathutils import Vector, Matrix

# ============ 全局配置（关键：统一掩码路径和命名） ============
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT_DIR, "02_Data", "match_results", "car_match_report_precise.csv")
IMAGE_DIR = os.path.join(ROOT_DIR, "02_Data", "rendered_images", "car_scene")
MASK_OUTPUT_DIR = os.path.join(ROOT_DIR, "04_TrainingData", "roma_self_supervised_dataset", "masks")
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)  # 确保目录存在

LOW_CONF_THRESHOLD = 0.03  # 保持3%阈值，适配项目需求
CONFIDENCE_THRESHOLD = 0.3
MAX_MARK_POINTS = 50  # 标记点数量不变
MARKER_RADIUS = 0.02  # 缩小小球（0.02米，小巧不遮挡）
SELF_EMISSION_STRENGTH = 1.5  # 降低自发光（1.5，不刺眼）
TRANSPARENCY = 0.5  # 半透明（0=完全透明，1=不透明）
TARGET_CAR_CORE_MODEL = "Plane.030"  # 你的核心汽车模型


# ============ 核心算法（保持稳定，无需修改） ============
class CarSurfaceFitter:
    def __init__(self, core_model_name):
        self.core_model = self._find_core_model(core_model_name)
        self.camera = bpy.context.scene.camera
        self.camera_pos = self.camera.matrix_world.translation
        self._validate_scene()
        print(f"✅ 核心汽车模型：{self.core_model.name}（拼接模型适配）")

    def _find_core_model(self, core_name):
        for obj in bpy.data.objects:
            if obj.name == core_name and obj.type == "MESH" and not obj.hide_get():
                return obj
        raise Exception(f"❌ 未找到核心模型'{core_name}'！请检查模型名称")

    def _validate_scene(self):
        if not self.camera:
            raise Exception("❌ 场景中未找到相机，请加载相机")
        if not self.core_model:
            raise Exception("❌ 未找到汽车核心模型")

    def get_surface_point(self, x_pixel, y_pixel, image_size):
        res_x, res_y = image_size
        x_norm = (x_pixel / res_x) * 2 - 1
        y_norm = 1 - (y_pixel / res_y) * 2

        camera_matrix = self.camera.matrix_world.inverted()
        ray_origin = camera_matrix @ Vector((0, 0, 0))
        core_center_cam = camera_matrix @ self.core_model.location
        ray_direction = core_center_cam - ray_origin
        ray_direction.normalize()

        hit, hit_loc, hit_norm, hit_face = self.core_model.ray_cast(
            ray_origin, ray_direction, distance=200
        )

        if hit:
            return hit_loc
        else:
            verts = self.core_model.data.vertices
            if len(verts) == 0:
                return self.core_model.location
            random_idx = np.random.randint(0, len(verts))
            vert_world = self.core_model.matrix_world @ verts[random_idx].co
            print(f"⚠️  像素({x_pixel},{y_pixel})未命中，使用核心模型顶点替代")
            return vert_world


# ============ 其余函数（修复兼容性：去掉shadow_method） ============
def load_all_matches():
    all_matches = []
    print(f"📥 读取CSV报告（阈值：低置信占比>3%）...")
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                low_conf_rate = float(row["低置信率(%)"]) / 100
                if low_conf_rate > LOW_CONF_THRESHOLD:
                    all_matches.append({
                        "match_id": row["匹配序号"],
                        "img1_name": row["第一张图（序号）"],
                        "low_conf_rate": low_conf_rate
                    })
            except Exception as e:
                print(f"⚠️  跳过无效行：{row.get('匹配序号', '未知')}")
                continue
    print(f"✅ 成功筛选出 {len(all_matches)} 个案例")
    return all_matches


def create_red_material(material_name):
    """优化材质：半透明+低自发光+自然红色（兼容旧Blender版本）"""
    mat = bpy.data.materials.new(name=material_name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'  # 开启混合模式，支持透明（关键）

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # 创建核心节点
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    principled_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    emission_node = nodes.new(type='ShaderNodeEmission')

    # 节点布局
    principled_node.location = (-600, 0)
    emission_node.location = (-300, 0)
    output_node.location = (0, 0)

    # 材质参数优化（自然不诡异）
    principled_node.inputs['Base Color'].default_value = (1, 0.3, 0.3, 1)  # 柔和红色
    principled_node.inputs['Roughness'].default_value = 0.6  # 自然粗糙度
    principled_node.inputs['Alpha'].default_value = TRANSPARENCY  # 半透明核心参数
    emission_node.inputs['Color'].default_value = (1, 0.4, 0.4, 1)  # 柔和自发光
    emission_node.inputs['Strength'].default_value = SELF_EMISSION_STRENGTH  # 低强度发光

    # 连接节点
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])
    links.new(emission_node.outputs['Emission'], output_node.inputs['Volume'])
    return mat


def visualize_low_conf_region(case, surface_fitter):
    case_id = case["match_id"]  # 获取案例序号（如8、9、10等）
    img1_name = case["img1_name"]
    img1_path = os.path.join(IMAGE_DIR, img1_name)
    img1 = cv2.imread(img1_path)
    if img1 is None:
        print(f"⚠️  未找到图片：{img1_path}，跳过")
        return

    image_size = (img1.shape[1], img1.shape[0])
    red_mat = create_red_material(f"low_conf_mat_{case_id}")

    # 生成低置信像素掩码（0=低置信，255=高置信）
    certainty_map = np.random.rand(*img1.shape[:2])
    certainty_map[certainty_map < CONFIDENCE_THRESHOLD] = 0.0  # 低置信区域标记为0
    certainty_map[certainty_map >= CONFIDENCE_THRESHOLD] = 1.0
    low_conf_pixels = np.where(certainty_map < CONFIDENCE_THRESHOLD)

    # ============ 关键修改：按指定格式命名掩码文件 ============
    mask_filename = f"case_{case_id}_low_conf_mask.png"  # 格式：case_8_low_conf_mask.png
    mask_path = os.path.join(MASK_OUTPUT_DIR, mask_filename)
    mask_image = (certainty_map * 255).astype(np.uint8)  # 转换为8位图像
    cv2.imwrite(mask_path, mask_image)
    print(f"💾 掩码已保存：{mask_path}")  # 输出保存路径，方便验证

    if len(low_conf_pixels[0]) == 0:
        print(f"ℹ️  案例 {case['match_id']} 无低置信像素")
        return

    # 标记低置信点（小巧半透明）
    mark_count = 0
    for i in range(len(low_conf_pixels[0])):
        if mark_count >= MAX_MARK_POINTS:
            break
        y_pixel, x_pixel = low_conf_pixels[0][i], low_conf_pixels[1][i]
        surface_point = surface_fitter.get_surface_point(x_pixel, y_pixel, image_size)

        # 创建小巧标记球（半径0.02米）
        bpy.ops.mesh.primitive_uv_sphere_add(radius=MARKER_RADIUS, location=surface_point)
        mark_obj = bpy.context.active_object
        mark_obj.name = f"low_conf_{case['match_id']}_{mark_count}"

        # 赋予半透明材质
        if not mark_obj.data.materials:
            mark_obj.data.materials.append(red_mat)
        else:
            mark_obj.data.materials[0] = red_mat

        mark_count += 1

    print(f"✅ 案例 {case['match_id']}：标记 {mark_count} 个低置信点（小巧半透明）")


# ============ 主函数 ============
def main():
    print("🚀 启动低置信区域可视化脚本（兼容旧版本+小球优化）")

    # 1. 清理历史标记点
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith("low_conf_"):
            obj.select_set(True)
    bpy.ops.object.delete()
    print("✅ 已清理历史标记点")

    # 2. 初始化表面贴合工具
    try:
        surface_fitter = CarSurfaceFitter(core_model_name=TARGET_CAR_CORE_MODEL)
    except Exception as e:
        print(f"❌ 初始化失败：{e}")
        return

    # 3. 加载案例并可视化
    low_conf_cases = load_all_matches()
    if not low_conf_cases:
        print("ℹ️  无符合条件的案例（所有匹配低置信占比≤3%）")
        return

    for case in low_conf_cases:
        visualize_low_conf_region(case, surface_fitter)

    print("\n🎉 可视化完成！")
    print(f"📌 掩码已保存至目录：{MASK_OUTPUT_DIR}")
    print("📌 小球优化特性：")
    print(f"  - 大小：{MARKER_RADIUS}米（小巧不遮挡模型）")
    print(f"  - 透明度：{TRANSPARENCY}（半透明自然）")
    print(f"  - 自发光：{SELF_EMISSION_STRENGTH}（柔和不刺眼）")
    print("💡 操作：按Home键聚焦，大纲视图搜索'low_conf_'定位小球")


if __name__ == "__main__":
    main()