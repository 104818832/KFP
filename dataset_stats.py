"""
=========================================================
KFP-PDD Dataset Statistics Tool

Functions
---------
1. Image number statistics
2. YOLO label statistics
3. Class distribution
4. Excel report
5. Visualization

Author: KFP-YOLO
=========================================================
"""


import os
from pathlib import Path
from collections import Counter

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================


DATASET_DIR = Path(
    r"KFP-PDD"
)


SAVE_DIR = Path(
    r"statistics"
)


SAVE_DIR.mkdir(
    exist_ok=True
)


SPLITS = [
    "train",
    "val",
    "test"
]


IMG_EXT = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
]


# KFP-PDD 11 classes

CLASS_NAMES = {

    0:"Pear Leaf Healthy",

    1:"Pear Leaf Chlorosis Disease",

    2:"Pear Leaf Spot Disease",

    3:"Pear Leaf Fire Blight",

    4:"Pear Leaf Insect Damage",

    5:"Pear Leaf Infestation by Pear Gall Midge",

    6:"Pear Fruit Healthy",

    7:"Pear Fruit Grapholita molesta Infestation",

    8:"Pear Fruit Fire Blight",

    9:"Pear Flower Healthy",

    10:"Pear Flower Fire Blight"

}


# =========================================================
# Read YOLO label
# =========================================================


def read_label(label_path):

    classes=[]


    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:


        for line in f:


            if line.strip()=="":

                continue


            cls=int(
                line.split()[0]
            )


            classes.append(
                cls
            )


    return classes



# =========================================================
# Analyze one split
# =========================================================


def analyze_split(split):


    image_dir=(
        DATASET_DIR
        /
        "images"
        /
        split
    )


    label_dir=(
        DATASET_DIR
        /
        "labels"
        /
        split
    )


    images=[
        x for x in image_dir.iterdir()
        if x.suffix.lower()
        in IMG_EXT
    ]


    class_counter=Counter()


    image_counter=0

    object_counter=0


    missing_labels=[]


    image_sizes=[]


    for img in tqdm(
        images,
        desc=split
    ):


        image_counter+=1


        # image size

        try:

            im=cv2.imread(
                str(img)
            )


            if im is not None:

                h,w=im.shape[:2]

                image_sizes.append(
                    (w,h)
                )

        except:

            pass



        label=(
            label_dir
            /
            (img.stem+".txt")
        )


        if not label.exists():

            missing_labels.append(
                img.name
            )

            continue



        classes=read_label(
            label
        )


        object_counter+=len(
            classes
        )


        class_counter.update(
            classes
        )


    return {


        "images":image_counter,


        "objects":object_counter,


        "classes":class_counter,


        "missing":missing_labels,


        "sizes":image_sizes

    }



# =========================================================
# Main
# =========================================================


print("="*60)

print(
    "KFP-PDD Dataset Statistics"
)

print("="*60)



results={}


for split in SPLITS:


    results[split]=analyze_split(
        split
    )



# =========================================================
# Generate table
# =========================================================


rows=[]


for cls,name in CLASS_NAMES.items():


    row={

        "Class ID":cls,

        "Class Name":name,

    }


    total=0


    for split in SPLITS:


        num=(
            results[split]
            ["classes"]
            [cls]
        )


        row[split]=num


        total+=num


    row["Total"]=total


    rows.append(row)



df=pd.DataFrame(
    rows
)



df.to_excel(
    SAVE_DIR/
    "class_distribution.xlsx",
    index=False
)



# =========================================================
# Generate TXT
# =========================================================


txt_file=(
    SAVE_DIR/
    "dataset_summary.txt"
)



with open(
    txt_file,
    "w",
    encoding="utf-8"
) as f:


    f.write(
        "KFP-PDD Dataset Statistics\n"
    )


    f.write(
        "="*60+"\n\n"
    )


    for split in SPLITS:


        f.write(
            f"{split.upper()}\n"
        )


        f.write(
            "-"*40+"\n"
        )


        f.write(
            f"Images : "
            f"{results[split]['images']}\n"
        )


        f.write(
            f"Objects: "
            f"{results[split]['objects']}\n\n"
        )



    f.write(
        "\nClass Distribution\n"
    )


    f.write(
        "-"*40+"\n"
    )


    for _,row in df.iterrows():


        f.write(

            f"{row['Class ID']} "
            f"{row['Class Name']} : "
            f"{row['Total']}\n"

        )



# =========================================================
# Draw chart
# =========================================================


plt.figure(
    figsize=(12,6)
)


plt.bar(
    df["Class ID"],
    df["Total"]
)


plt.xlabel(
    "Class ID"
)


plt.ylabel(
    "Object Number"
)


plt.title(
    "KFP-PDD Class Distribution"
)


plt.xticks(
    df["Class ID"]
)


plt.tight_layout()


plt.savefig(
    SAVE_DIR/
    "class_distribution.png",
    dpi=300
)


plt.close()



# =========================================================
# Print result
# =========================================================


print("\nDataset Summary")

print("-"*60)


for split in SPLITS:


    print(

        f"{split}: "
        f"{results[split]['images']} images | "
        f"{results[split]['objects']} objects"

    )


print("\nClass Distribution")

print(df)



print("\nMissing Labels")

for split in SPLITS:

    print(
        split,
        len(
            results[split]["missing"]
        )
    )


print("="*60)

print(
    "Statistics Finished"
)

print(
    "Saved to:",
    SAVE_DIR
)

print("="*60)