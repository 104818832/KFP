\# KFP-YOLO



Official source code for the paper:



\*\*KFP-YOLO: A Lightweight Detection Model for Korla Fragrant Pear Diseases and Pests Detection toward Edge Deployment\*\*



\---



\## Overview



KFP-YOLO is a lightweight object detection model developed based on the YOLO26n framework for Korla fragrant pear diseases and pests detection.



Compared with the baseline YOLO26n model, KFP-YOLO introduces three lightweight modules:



\- \*\*ADown\*\*: Lightweight downsampling module

\- \*\*C3-PD\*\*: Lightweight feature extraction module based on Partial Convolution

\- \*\*CFA\*\*: Coordinate Feature Attention module



The proposed model is designed for accurate and real-time deployment on edge devices.



\---



\# Directory Structure



```

KFP-YOLO-source

│

├── datasets/                 # Dataset configuration

├── models/                   # Model configuration files

├── onnx/                     # Exported ONNX models

├── ultralytics/              # Modified Ultralytics source code

├── weights/                  # Trained model weights

│

├── KFP.yaml                  # Dataset configuration

├── train.py                  # Training script

├── val.py                    # Validation script

├── predict.py                # Inference script

│

├── requirements.txt          # Python dependencies

└── README.md

```



\---



\# Experimental Environment



| Item | Version |

|------|---------|

| Python | 3.12.12 |

| PyTorch | 2.5.1 |

| CUDA | 12.1 |

| Ultralytics | 8.4.0 |

| ONNX | 1.16.0 |

| THOP | 2.0.18 |



\---



\# Installation



Install all required packages.



```bash

pip install -r requirements.txt

```



\---



\# Dataset



Prepare the dataset in YOLO format.



```

datasets/

│

├── images/

│   ├── train/

│   ├── val/

│   └── test/

│

└── labels/

&#x20;   ├── train/

&#x20;   ├── val/

&#x20;   └── test/

```



Modify the dataset path in `KFP.yaml` before training.



\---



\# Training



Run the training script.



```bash

python train.py

```



Recommended training settings:



\- Image size: 640 × 640

\- Epochs: 300

\- Batch size: 32

\- Optimizer: SGD

\- Initial learning rate: 0.01



\---



\# Validation



Run validation.



```bash

python val.py

```



Evaluation metrics include:



\- Precision

\- Recall

\- mAP@0.5

\- mAP@0.5:0.95



\---



\# Inference



Run prediction.



```bash

python predict.py

```



\---



\# Model Weights



Place the trained model weights in:



```

weights/

```



Example:



```

weights/best.pt

```



\---



\# ONNX Export



Export the trained model to ONNX format.



```bash

python export.py

```



The exported model will be saved in the `onnx/` directory.



\---



\# Notes



This repository is developed based on the Ultralytics framework.



Only the modified modules and configurations related to KFP-YOLO are included.



\---



\# Citation



If you use this repository in your research, please cite the corresponding paper.



```

KFP-YOLO: A Lightweight Detection Model for Korla Fragrant Pear Diseases and Pests Detection toward Edge Deployment.

```



\---



\# License



This project is released for academic research purposes only.

