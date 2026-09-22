import os
import sys
import glob
import argparse
import traceback
import cv2
import torch
from pathlib import Path

# ==============================================================================
# 1. 导入服务端模块 (若你的服务端文件名不是 server.py，请修改此处的 `server`)
# ==============================================================================
SERVER_MODULE_NAME = "run_reloc_server"  # <--- 请根据实际 Server 文件名修改（不要写 .py 后缀）

try:
    server_module = __import__(SERVER_MODULE_NAME)
    build_service = server_module.build_service
    make_server_handler = server_module.make_server_handler
    print(f"✅ 成功导入服务构建函数: {SERVER_MODULE_NAME}.py")
except ImportError:
    print(f"❌ 导入失败！在当前目录下未找到 `{SERVER_MODULE_NAME}.py` 文件。")
    print("请修改脚本开头的 `SERVER_MODULE_NAME = '你的服务端文件名'`。")
    sys.exit(1)


def run_local_test():
    dataset_dir = "../test"
    map_dir = os.path.join(dataset_dir, "hloc_map")
    color_dir = os.path.join(dataset_dir, "scan1/color")

    print("\n" + "=" * 60)
    print("🚀 开始本地 RelocService 本地完整流程自测...")
    print("=" * 60)

    # -------------------------------------------------------------
    # Step 1: 检查本地图片数据
    # -------------------------------------------------------------
    if not os.path.exists(color_dir):
        print(f"❌ 目录不存在: {os.path.abspath(color_dir)}")
        return

    image_files = sorted(glob.glob(os.path.join(color_dir, "*.*")))
    if not image_files:
        print(f"❌ 在 {color_dir} 下未找到任何图片文件！")
        return

    test_image_path = image_files[0]
    print(f"✅ 找到 {len(image_files)} 张本地图，选取测试图: {os.path.basename(test_image_path)}")

    # -------------------------------------------------------------
    # Step 2: 构建测试所需的 service_args 参数对象
    # -------------------------------------------------------------
    print("\n[Step 1/3] 配置并构造 service_args...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_cache_device = device

    # 模拟命令行解析得到的参数结构体
    service_args = argparse.Namespace(
        root_dir=Path(dataset_dir),
        map_dir=Path(map_dir),
        image_path=Path(color_dir),
        retrieval_conf="megaloc",           # 检索配置名，按实际改
        local_feature_conf="superpoint_inloc", # 局部特征配置名，按实际改
        matcher_conf="superglue",          # 匹配器配置名，按实际改
        num_retrieval=20,
        num_match_db=8,
        min_correspondences=256,
        min_matched_db=4,
        ransac_thresh=12.0,
        train_list=None,
        default_camera_model="PINHOLE",
        default_camera_params="752.560669,751.958862,641.847534,357.389160",
        fallback_to_nearest_db=True,
        retvis=False,
        rotation_augmentation=False,
    )

    # -------------------------------------------------------------
    # Step 3: 实例化 RelocService 与 Handler
    # -------------------------------------------------------------
    print("\n[Step 2/3] 正在通过 build_service 实例化 RelocService...")
    try:
        service_instance = build_service(
            service_args=service_args,
            device=device,
            local_cache_device=local_cache_device,
            map_name=None,
        )
        print("✅ RelocService 实例化成功！")

        # 构造顶层 handler 处理函数 (与 netcall 上调用的 handler 一致)
        handler = make_server_handler(service=service_instance, manager=None, log_server_response=True)
        print("✅ make_server_handler 包装成功！")

    except Exception:
        print("\n❌ 服务构建阶段抛出异常！详细堆栈信息：")
        traceback.print_exc()
        return

    # -------------------------------------------------------------
    # Step 4: 读取测试图像并构造请求包 objs
    # -------------------------------------------------------------
    print("\n[Step 3/3] 准备请求数据并执行重定位推理...")
    image_bgr = cv2.imread(test_image_path)
    if image_bgr is None:
        print(f"❌ 读取图像失败: {test_image_path}")
        return

    # 模拟网络接收到的反序列化数据字典
    objs = {
        "image": image_bgr,
        "camera_params": "752.560669,751.958862,641.847534,357.389160",
        "db_name": os.path.basename(test_image_path),
    }

    try:
        # 直接调用 handler 模拟服务端接收网络请求的整套动作
        response = handler(objs)

        print("\n" + "=" * 60)
        print("🎉 测试通过！服务端成功处理请求，返回最小化响应：")
        print("=" * 60)
        print("Response 内容:", response)

    except Exception:
        print("\n" + "!" * 60)
        print("💥 localize / 推理执行过程中捕获到异常！详细堆栈信息：")
        print("!" * 60)
        traceback.print_exc()


if __name__ == "__main__":
    run_local_test()