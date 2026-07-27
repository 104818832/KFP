import os
from ultralytics import YOLO

# ===== 路径配置 =====
project_root = r"E:\xuzhuoyang\KFPS\yolo26nkfps"
save_root = r"E:\xuzhuoyang\KFPS\onnx"

os.makedirs(save_root, exist_ok=True)


def export_all_to_onnx():
    model_paths = []

    # 1. 搜索所有 best.pt
    for root, dirs, files in os.walk(project_root):
        if 'best.pt' in files:
            model_paths.append(os.path.join(root, 'best.pt'))

    print(f"🚀 共找到 {len(model_paths)} 个模型，开始导出 ONNX...\n")

    for path in model_paths:
        try:
            # 2. 提取模型名（如 yolo11n）
            model_name = os.path.basename(os.path.dirname(os.path.dirname(path)))

            print(f"🔄 正在导出: {model_name}")

            # 3. 加载模型
            model = YOLO(path)

            # 4. 导出 ONNX（注意：会输出到 weights 目录）
            model.export(
                format="onnx",
                imgsz=640,
                simplify=True,      # 推荐
                dynamic=False,      # TensorRT更稳
                opset=12            # 兼容性最好
            )

            # 5. 从 weights 目录获取 ONNX
            weights_dir = os.path.dirname(path)  # .../weights/
            onnx_path = os.path.join(weights_dir, "best.onnx")

            # 6. 目标路径（统一命名）
            new_path = os.path.join(save_root, f"{model_name}.onnx")

            if os.path.exists(onnx_path):
                # 覆盖旧文件
                if os.path.exists(new_path):
                    os.remove(new_path)

                os.rename(onnx_path, new_path)

                print(f"✅ 完成: {model_name} → {model_name}.onnx\n")
            else:
                print(f"❌ 未找到 ONNX 文件: {onnx_path}\n")

        except Exception as e:
            print(f"❌ {model_name} 导出失败: {e}\n")

    print("🎉 全部 ONNX 导出完成！")


if __name__ == "__main__":
    export_all_to_onnx()