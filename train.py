"""
KFP-YOLO Training Script

Paper:
KFP-YOLO: A Lightweight Detection Model for Korla Fragrant Pear Diseases and Pests Detection toward Edge Deployment
"""

from pathlib import Path
import gc
import traceback

import torch
from ultralytics import YOLO

# =====================================================
# Configuration
# =====================================================

# Dataset configuration
DATA_PATH = "KFP.yaml"

# Model configuration (or pretrained weight)
MODEL_PATH = "models/KFP-YOLO.yaml"

# Output directory
PROJECT = "runs/train"
NAME = "KFP-YOLO"

# Training parameters
EPOCHS = 300
IMGSZ = 640
BATCH = 32
WORKERS = 4

SEED = 0

# =====================================================


def train():

    project_dir = Path(PROJECT) / NAME
    last_weight = project_dir / "weights" / "last.pt"

    resume = last_weight.exists()

    print("=" * 60)
    print("KFP-YOLO Training")
    print("=" * 60)
    print(f"Dataset : {DATA_PATH}")
    print(f"Model   : {MODEL_PATH}")
    print(f"Output  : {project_dir}")
    print(f"Resume  : {resume}")
    print("=" * 60)

    try:

        model = YOLO(str(last_weight) if resume else MODEL_PATH)

        model.train(

            # Dataset
            data=DATA_PATH,

            # Training
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            workers=WORKERS,

            # Optimizer
            optimizer="SGD",

            # Learning rate
            lr0=0.01,
            lrf=0.01,

            # Random seed
            seed=SEED,
            deterministic=False,

            # Mixed precision
            amp=True,

            # Cache
            cache=False,

            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            translate=0.1,
            scale=0.5,
            fliplr=0.5,
            mosaic=1.0,

            # Save
            save_period=5,

            # Output
            project=PROJECT,
            name=NAME,
            exist_ok=True,

            # Resume
            resume=resume,
        )

        print("\nTraining completed successfully!")

    except Exception:

        print("\nTraining failed!")

        traceback.print_exc()

    finally:

        if "model" in locals():
            del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        print("GPU memory released.")


if __name__ == "__main__":
    train()