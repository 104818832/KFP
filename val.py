"""
KFP-YOLO Validation Script

Paper:
KFP-YOLO: A Lightweight Detection Model for Korla Fragrant Pear Diseases and Pests Detection toward Edge Deployment
"""

from pathlib import Path
import traceback

from ultralytics import YOLO

# =====================================================
# Configuration
# =====================================================

# Dataset configuration
DATA_PATH = "KFP.yaml"

# Trained model
WEIGHTS = "weights/best.pt"

# Output directory
PROJECT = "runs/val"
NAME = "KFP-YOLO"

# Validation parameters
IMGSZ = 640
BATCH = 32
DEVICE = 0      # GPU:0, CPU: device="cpu"

# =====================================================


def validate():

    print("=" * 60)
    print("KFP-YOLO Validation")
    print("=" * 60)
    print(f"Dataset : {DATA_PATH}")
    print(f"Weights : {WEIGHTS}")
    print("=" * 60)

    if not Path(WEIGHTS).exists():
        print(f"\nError: Model weight not found: {WEIGHTS}")
        return

    try:

        model = YOLO(WEIGHTS)

        results = model.val(

            data=DATA_PATH,

            imgsz=IMGSZ,

            batch=BATCH,

            device=DEVICE,

            project=PROJECT,

            name=NAME,

            exist_ok=True,

            plots=True,

            save_json=False,

            verbose=True,

        )

        print("\nValidation completed successfully!")

        # Print evaluation metrics
        try:
            print("\n========== Evaluation Results ==========")
            print(f"Precision (P)     : {results.box.mp:.4f}")
            print(f"Recall (R)        : {results.box.mr:.4f}")
            print(f"mAP@0.5           : {results.box.map50:.4f}")
            print(f"mAP@0.5:0.95      : {results.box.map:.4f}")
            print("========================================")
        except Exception:
            print("\nMetrics have been saved to the output directory.")

    except Exception:

        print("\nValidation failed!")

        traceback.print_exc()


if __name__ == "__main__":

    validate()