"""
KFP-YOLO Prediction Script

Paper:
KFP-YOLO: A Lightweight Detection Model for Korla Fragrant Pear Diseases and Pests Detection toward Edge Deployment
"""

from pathlib import Path
import traceback

from ultralytics import YOLO

# =====================================================
# Configuration
# =====================================================

# Trained model
WEIGHTS = "weights/best.pt"

# Input source
# Examples:
# source = "test.jpg"
# source = "test.mp4"
# source = "datasets/images/test"
# source = 0
SOURCE = "test.jpg"

# Output
PROJECT = "runs/predict"
NAME = "KFP-YOLO"

# Inference parameters
IMGSZ = 640
CONF = 0.25
IOU = 0.45
DEVICE = 0

# =====================================================


def predict():

    print("=" * 60)
    print("KFP-YOLO Prediction")
    print("=" * 60)
    print(f"Weights : {WEIGHTS}")
    print(f"Source  : {SOURCE}")
    print("=" * 60)

    if not Path(WEIGHTS).exists():
        print(f"\nError: Weight file not found: {WEIGHTS}")
        return

    try:

        model = YOLO(WEIGHTS)

        model.predict(

            source=SOURCE,

            imgsz=IMGSZ,

            conf=CONF,

            iou=IOU,

            device=DEVICE,

            save=True,

            save_txt=False,

            save_conf=False,

            show=False,

            project=PROJECT,

            name=NAME,

            exist_ok=True,

            verbose=True

        )

        print("\nPrediction completed successfully!")

        print(f"Results saved to: {PROJECT}/{NAME}")

    except Exception:

        print("\nPrediction failed!")

        traceback.print_exc()


if __name__ == "__main__":

    predict()