import os
import sys
import glob
import traceback
import cv2
import numpy as np
import torch

# 1. 导入你的类 (请将 relocalization_server 改为你实际的文件名，RelocalizationServer 改为实际类名)
try:
    from relocalization_server import RelocalizationServer
except ImportError:
    print("❌ 导入失败！请检查文件路径或类名。")
    print("请确认 `relocalization_server.py` 在当前目录下，且类名匹配。")
    sys.exit(1)


def run_test():
    dataset_dir = "../test/scan1"
    color_dir = os.path.join(dataset_dir, "color")
    camera_info_path = os.path.join(dataset_dir, "camera_info.txt")

    print("=" * 60)
    print("🚀 开始本地重定位流程调试与测试...")
    print("=" * 60)

    # -------------------------------------------------------------
    # 步骤 1: 检查本地数据集文件是否存在
    # -------------------------------------------------------------
    if not os.path.exists(color_dir):
        print(f"❌ 目录不存在: {os.path.abspath(color_dir)}")
        return

    # 获取 color 文件夹下的图片列表
    image_files = sorted(glob.glob(os.path.join(color_dir, "*.*")))
    if not image_files:
        print(f"❌ 在 {color_dir} 下未找到任何图片文件！")
        return
    
    print(f"✅ 找到 {len(image_files)} 张数据库图片。")
    test_db_filename = os.path.basename(image_files[0])  # 挑选第一张作为 DB 图测试
    test_query_img_path = image_files[-1] if len(image_files) > 1 else image_files[0] # 挑选另一张作为 Query 图
    
    print(f"   ├─ 选中测试 DB 图片名: {test_db_filename}")
    print(f"   └─ 选中测试 Query 图路径: {test_query_img_path}")

    # -------------------------------------------------------------
    # 步骤 2: 实例化服务对象
    # -------------------------------------------------------------
    print("\n[Step 1/4] 正在初始化 Server 对象...")
    try:
        # 如果你的类 __init__ 接受 base_dir，请在此传入
        server = RelocalizationServer(base_dir=dataset_dir)
        print("✅ Server 初始化成功。")
    except Exception:
        print("❌ Server 初始化失败，详细堆栈如下：")
        traceback.print_exc()
        return

    # -------------------------------------------------------------
    # 步骤 3: 专门测试内参解析 & 灰度图读取
    # -------------------------------------------------------------
    print("\n[Step 2/4] 测试 get_db_camera_params 和 get_db_image_gray...")
    try:
        db_params = server.get_db_camera_params(test_db_filename)
        print(f"✅ 成功提取内参: {db_params}")

        db_gray_img = server.get_db_image_gray(test_db_filename)
        print(f"✅ 成功读取库图灰度图，图像 Shape: {db_gray_img.shape}, dtype: {db_gray_img.dtype}")
    except Exception:
        print("❌ 读取内参或库图失败，详细堆栈如下：")
        traceback.print_exc()
        return

    # -------------------------------------------------------------
    # 步骤 4: 构造模拟 Request 数据包
    # -------------------------------------------------------------
    print("\n[Step 3/4] 构造测试 Request 数据包...")
    # 模拟真实 Query 图数据 (读取 BGR 数组)
    query_bgr = cv2.imread(test_query_img_path)
    if query_bgr is None:
        print(f"❌ 读取 Query 图片失败: {test_query_img_path}")
        return

    # 模拟客户端发送的内参字符串或列表
    mock_camera_params = "752.560669,751.958862,641.847534,357.389160"

    mock_request = {
        "image": query_bgr,               # 传入 query 图像 (numpy 数组)
        "camera_params": mock_camera_params,  # 传入字符串形式内参，测试切割兼容性
        "db_name": test_db_filename       # 目标 DB 图片名
    }
    print("✅ Request 数据构造完成。")

    # -------------------------------------------------------------
    # 步骤 5: 调用核心定位函数 (如 localize / process)
    # -------------------------------------------------------------
    print("\n[Step 4/4] 执行重定位核心算法流程...")
    try:
        # 假设你的主入口函数名为 localize，如果叫 process_request 或其他名字请修改此处
        if hasattr(server, 'localize'):
            result = server.localize(mock_request)
        elif hasattr(server, 'process'):
            result = server.process(mock_request)
        else:
            print("⚠️ 未找到 localize 或 process 入口方法，请确认你的调用方法名。")
            return

        print("\n" + "=" * 60)
        print("🎉 测试通过！推理成功返回！")
        print("=" * 60)
        print("返回结果:", result)

    except Exception:
        print("\n" + "!" * 60)
        print("💥 推理过程中捕获到异常！错误堆栈信息如下：")
        print("!" * 60)
        traceback.print_exc()


if __name__ == "__main__":
    run_test()