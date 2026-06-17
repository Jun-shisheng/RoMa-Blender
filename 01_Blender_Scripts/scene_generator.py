# -*- coding: utf-8 -*-
import bpy
import os
import time
import traceback
import gc

# ============ 路径设置 ============
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "02_Data")
RENDER_DIR = os.path.join(DATA_DIR, "rendered_images")
BLENDER_PROJ_DIR = os.path.join(DATA_DIR, "blender_projects")
LOCAL_ASSETS_DIR = os.path.join(DATA_DIR, "local_assets")

for d in [RENDER_DIR, BLENDER_PROJ_DIR, LOCAL_ASSETS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============ 场景配置 ============
scenes = [
    {
        "name": "car_scene",
        "asset_folder": "car",
        "time_interval": 1,  # 每隔1秒渲染1张
        "max_render_frames": 30  # 最大渲染帧数（防止无限渲染，可调整）
    }
]


# ============ 工具函数 ============
def clean_up():
    """彻底清理当前场景"""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for col in [bpy.data.meshes, bpy.data.materials, bpy.data.images]:
        for item in list(col):
            try:
                col.remove(item)
            except Exception:
                pass
    gc.collect()
    print("🔧 已彻底清理场景资源")


def import_model(folder):
    """导入模型但不修改其原有设置"""
    path = os.path.join(LOCAL_ASSETS_DIR, folder)
    # 查找包含car且后缀为.blend的文件
    files = [f for f in os.listdir(path) if f.lower().endswith('.blend') and 'car' in f.lower()]
    if not files:
        raise FileNotFoundError(f"在 {path} 目录下未找到包含'car'的.blend文件")

    file = os.path.join(path, files[0])
    print(f"🔍 正在加载模型文件: {file}")

    # 导入.blend文件中的所有对象（保留原有配置）
    with bpy.data.libraries.load(file, link=False) as (src, dst):
        dst.objects = src.objects

    # 将所有对象链接到当前场景
    imported_objects = []
    for obj in dst.objects:
        if obj:
            bpy.context.collection.objects.link(obj)
            imported_objects.append(obj)

    # 返回第一个网格对象作为主模型
    for obj in imported_objects:
        if obj.type == 'MESH':
            return obj
    return None if not imported_objects else imported_objects[0]


# ============ 场景生成主逻辑 ============
def process_scene(cfg):
    print(f"\n--- 正在处理场景: {cfg['name']} ---")
    try:
        clean_up()

        # 导入模型（保留所有预设参数）
        model = import_model(cfg["asset_folder"])
        if not model:
            raise Exception("未能加载汽车模型")
        print("✅ 模型加载完成（使用原有配置）")

        # 获取场景中已有的摄像机
        cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
        if not cameras:
            raise Exception("模型文件中未找到摄像机")
        bpy.context.scene.camera = cameras[0]
        print(f"✅ 已选中摄像机: {cameras[0].name}")

        # 获取Blender场景帧率（默认24fps，若你修改过会自动适配）
        scene_fps = bpy.context.scene.render.fps
        print(f"ℹ️ 当前场景帧率: {scene_fps} FPS")

        # 计算每隔1秒对应的帧数（比如24fps → 每24帧渲染1张）
        frame_interval = int(scene_fps * cfg["time_interval"])
        if frame_interval < 1:
            frame_interval = 1  # 防止帧率异常导致间隔为0
        print(f"ℹ️ 每隔 {cfg['time_interval']} 秒渲染1张 → 每 {frame_interval} 帧渲染1张")

        # 确定渲染范围（取场景动画长度或最大渲染帧数，取较小值）
        scene_end_frame = bpy.context.scene.frame_end
        render_end_frame = min(scene_end_frame, cfg["max_render_frames"])
        print(f"ℹ️ 渲染范围: 第0帧 ~ 第{render_end_frame}帧")

        # 创建渲染输出目录
        render_dir = os.path.join(RENDER_DIR, cfg["name"])
        os.makedirs(render_dir, exist_ok=True)

        # 按时间间隔渲染图片
        render_count = 0
        for frame in range(0, render_end_frame + 1, frame_interval):
            # 跳转到目标帧
            bpy.context.scene.frame_set(frame)

            # 计算当前帧对应的时间（秒）
            current_time = round(frame / scene_fps, 2)
            # 保存图片（文件名包含帧号和时间，便于后续对应）
            out_path = os.path.join(render_dir, f"frame_{frame}_time_{current_time}s.png")

            # 渲染并保存
            bpy.context.scene.render.filepath = out_path
            bpy.ops.render.render(write_still=True)

            print(f"✅ 已渲染: 帧{frame} → 时间{current_time}秒 → {out_path}")
            render_count += 1

        # 保存工程文件
        blend_path = os.path.join(BLENDER_PROJ_DIR, f"{cfg['name']}.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        print(f"✅ 已保存Blender工程文件: {blend_path}")
        print(f"\n📊 渲染完成统计: 共渲染 {render_count} 张图片")

    except Exception as e:
        print(f"❌ 场景 {cfg['name']} 渲染失败: {e}")
        traceback.print_exc()
    finally:
        clean_up()
        time.sleep(0.5)


# ============ 主入口 ============
if __name__ == "__main__":
    print("🚀 启动Blender汽车模型定时渲染脚本")
    # 禁用BlenderKit减少干扰
    if "blenderkit" in bpy.context.preferences.addons:
        bpy.ops.preferences.addon_disable(module="blenderkit")
    for cfg in scenes:
        process_scene(cfg)
    print("\n🎉 所有场景处理完成！")
    print(f"📁 渲染结果目录: {RENDER_DIR}")