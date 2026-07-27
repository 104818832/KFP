"""
=========================================================
KFP-PDD Data Augmentation

Indoor:
    U2-Net background removal

Outdoor:
    50% dust
    50% filter

YOLO label compatible
=========================================================
"""

import os
import cv2
import random
import shutil
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from rembg import remove


# ==============================
# Configuration
# ==============================

IMAGE_DIR = Path(
    r"KFP-PDD/images/train"
)

LABEL_DIR = Path(
    r"KFP-PDD/labels/train"
)


OUT_IMAGE = Path(
    r"KFP-PDD_aug/images"
)

OUT_LABEL = Path(
    r"KFP-PDD_aug/labels"
)


SEED = 0

random.seed(SEED)


IMG_EXT = [
    ".jpg",
    ".png",
    ".jpeg"
]


OUT_IMAGE.mkdir(
    parents=True,
    exist_ok=True
)

OUT_LABEL.mkdir(
    parents=True,
    exist_ok=True
)


# ==============================
# 判断室内图片
# ==============================

def is_indoor(path):

    """
    根据文件名判断

    可根据你的数据修改

    """

    name = path.stem.lower()

    keywords = [
        "indoor",
        "inside",
        "lab",
        "room"
    ]

    for k in keywords:

        if k in name:

            return True

    return False



# ==============================
# U2-Net 去背景
# ==============================

def u2_remove_bg(img_path):

    """

    rembg 默认U2-Net

    返回BGR图片

    """

    img = Image.open(
        img_path
    )

    result = remove(
        img
    )


    result = np.array(
        result
    )


    if result.shape[2]==4:

        alpha = result[:,:,3]

        rgb = result[:,:,:3]


        background = np.ones_like(
            rgb
        )*255


        alpha = (
            alpha[:,:,None]/255
        )


        result = (
            rgb*alpha+
            background*(1-alpha)
        )


    else:

        result=result


    return cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )



# ==============================
# 沙尘增强
# ==============================

def add_dust(img):

    h,w,_=img.shape


    noise=np.random.normal(
        180,
        40,
        (h,w,1)
    )


    mask=np.random.random(
        (h,w,1)
    )


    mask=mask>0.65


    result=np.where(
        mask,
        img*0.7+
        noise*0.3,
        img
    )


    return np.uint8(
        np.clip(
            result,
            0,
            255
        )
    )



# ==============================
# 滤波增强
# ==============================

def add_filter(img):


    mode=random.choice(
        [
            0,
            1,
            2
        ]
    )


    if mode==0:

        # 模糊

        img=cv2.GaussianBlur(
            img,
            (3,3),
            0
        )


    elif mode==1:

        # 锐化

        kernel=np.array(
            [
                [0,-1,0],
                [-1,5,-1],
                [0,-1,0]
            ]
        )

        img=cv2.filter2D(
            img,
            -1,
            kernel
        )


    else:

        # 亮度

        factor=random.uniform(
            0.7,
            1.3
        )

        img=np.clip(
            img*factor,
            0,
            255
        )


    return np.uint8(img)



# ==============================
# 保存标签
# ==============================

def copy_label(
        label,
        name
):

    shutil.copy2(
        label,
        OUT_LABEL/name
    )



# ==============================
# 主程序
# ==============================


images=list(
    IMAGE_DIR.iterdir()
)


images=[
    x for x in images
    if x.suffix.lower()
    in IMG_EXT
]


print(
    "Images:",
    len(images)
)



for img_path in tqdm(images):


    label_path=(
        LABEL_DIR/
        (img_path.stem+".txt")
    )


    if not label_path.exists():

        continue



    img=cv2.imread(
        str(img_path)
    )


    # --------------------------
    # 室内 U2-Net
    # --------------------------

    if is_indoor(img_path):


        img=u2_remove_bg(
            img_path
        )


        name=(
            img_path.stem+
            "_u2.jpg"
        )


    # --------------------------
    # 室外增强
    # --------------------------

    else:


        if random.random()<0.5:

            img=add_dust(img)

            suffix="_dust"


        else:

            img=add_filter(img)

            suffix="_filter"


        name=(
            img_path.stem+
            suffix+
            ".jpg"
        )



    cv2.imwrite(
        str(
            OUT_IMAGE/name
        ),
        img
    )


    copy_label(
        label_path,
        name.replace(
            ".jpg",
            ".txt"
        )
    )



print("="*50)

print(
    "Augmentation Finished"
)

print(
    "Output:",
    OUT_IMAGE
)

print("="*50)