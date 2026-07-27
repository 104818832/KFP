from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ========================
# 1. 参数
# ========================
IMG_SIZE = 640

img_dir = r"D:\1"
save_dir = r"D:\vis_results"

cam_dir = os.path.join(save_dir, "cam")
compare_dir = os.path.join(save_dir, "compare")

os.makedirs(cam_dir, exist_ok=True)
os.makedirs(compare_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# 2. 加载模型
# ========================
model_base = YOLO(r"D:\KFPS\yolo\yolo26n\weights\best.pt")
model_improve = YOLO(r"D:\KFPS\kfps\yolo26n_adown_c3pdblock_cfa\weights\best.pt")

# ========================
# 3. 构建 CAM 模型（两个！！）
# ========================

def build_cam_model(yolo_model):
    model = yolo_model.model.model[:9]   # 🔥去掉Detect
    model.to(device)
    model.train()

    for p in model.parameters():
        p.requires_grad = True

    target_layers = [model[-1]]

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    return model, cam

model_cam_base, cam_base = build_cam_model(model_base)
model_cam_improve, cam_improve = build_cam_model(model_improve)

# ========================
# 4. Target
# ========================
class YOLOTarget:
    def __call__(self, model_output):
        return model_output.mean()

# ========================
# 5. CAM函数（通用）
# ========================
def process_image(img_path, cam, model_cam):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0

    input_tensor = torch.from_numpy(
        img_norm.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(device)

    input_tensor.requires_grad = True

    with torch.enable_grad():
        model_cam.zero_grad()

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[YOLOTarget()]
        )[0]

    cam_image = show_cam_on_image(img_norm, grayscale_cam, use_rgb=True)

    return cam_image

# ========================
# 6. 主流程
# ========================
for img_name in os.listdir(img_dir):
    img_path = os.path.join(img_dir, img_name)

    # ---------- 检测 ----------
    res1 = model_base(img_path, imgsz=IMG_SIZE, device=device)
    res2 = model_improve(img_path, imgsz=IMG_SIZE, device=device)

    base_path = os.path.join(save_dir, f"base_{img_name}")
    improve_path = os.path.join(save_dir, f"improve_{img_name}")

    res1[0].save(filename=base_path)
    res2[0].save(filename=improve_path)

    # ---------- CAM（两个模型） ----------
    cam_base_img = process_image(img_path, cam_base, model_cam_base)
    cam_improve_img = process_image(img_path, cam_improve, model_cam_improve)

    cv2.imwrite(os.path.join(cam_dir, f"cam_base_{img_name}"),
                cv2.cvtColor(cam_base_img, cv2.COLOR_RGB2BGR))

    cv2.imwrite(os.path.join(cam_dir, f"cam_improve_{img_name}"),
                cv2.cvtColor(cam_improve_img, cv2.COLOR_RGB2BGR))

    # ---------- 拼图（4图对比🔥） ----------
    base = cv2.imread(base_path)
    improve = cv2.imread(improve_path)

    base = cv2.resize(base, (IMG_SIZE, IMG_SIZE))
    improve = cv2.resize(improve, (IMG_SIZE, IMG_SIZE))
    cam_base_img = cv2.resize(cam_base_img, (IMG_SIZE, IMG_SIZE))
    cam_improve_img = cv2.resize(cam_improve_img, (IMG_SIZE, IMG_SIZE))

    # 👉 4图拼接（论文推荐）
    result = np.hstack((base, improve, cam_base_img, cam_improve_img))

    cv2.imwrite(os.path.join(compare_dir, f"compare_{img_name}"), result)

print("✅ 完成：双模型检测 + 双CAM + 四图对比")