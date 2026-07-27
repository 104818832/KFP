"""
=========================================================
KFP-YOLO Dataset Split Tool
Part 1

Functions
---------
1. Read YOLO dataset
2. Parse capture date from filename or EXIF
3. Read labels
4. Build image information database

Author : KFP-YOLO
=========================================================
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

# =========================================================
# Configuration
# =========================================================

# Original dataset
IMAGE_DIR = Path(r"dataset/images")
LABEL_DIR = Path(r"dataset/labels")

# Output dataset
OUTPUT_DIR = Path(r"KFP-PDD")

# Split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Random seed
SEED = 0

# Image extensions
IMAGE_SUFFIX = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
]

random.seed(SEED)

# =========================================================
# Utilities
# =========================================================

def create_output_dirs():

    folders = [

        "images/train",
        "images/val",
        "images/test",

        "labels/train",
        "labels/val",
        "labels/test",

    ]

    for folder in folders:

        (OUTPUT_DIR / folder).mkdir(
            parents=True,
            exist_ok=True
        )


# ---------------------------------------------------------

def find_image(label_name):

    """
    Find image according to label name.
    """

    stem = Path(label_name).stem

    for ext in IMAGE_SUFFIX:

        img = IMAGE_DIR / (stem + ext)

        if img.exists():
            return img

        img = IMAGE_DIR / (stem + ext.upper())

        if img.exists():
            return img

    return None


# ---------------------------------------------------------

def read_label(label_path):

    """
    Read YOLO label.
    Return:
        classes
        object_number
    """

    classes = []

    with open(label_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            cls = int(line.split()[0])

            classes.append(cls)

    return classes, len(classes)


# ---------------------------------------------------------

def get_exif_time(image_path):

    """
    Read shooting time from EXIF.
    """

    try:

        image = Image.open(image_path)

        exif = image._getexif()

        if exif is None:
            return None

        exif_data = {}

        for tag, value in exif.items():

            name = TAGS.get(tag, tag)

            exif_data[name] = value

        if "DateTimeOriginal" in exif_data:

            t = datetime.strptime(
                exif_data["DateTimeOriginal"],
                "%Y:%m:%d %H:%M:%S"
            )

            return t.date()

    except Exception:

        pass

    return None


# ---------------------------------------------------------

def get_filename_time(image_path):

    """
    Parse date from filename.

    Supported examples

    20250415_xxx.jpg
    IMG_20250415.jpg
    """

    name = image_path.stem

    digits = ""

    for c in name:

        if c.isdigit():

            digits += c

        else:

            if len(digits) >= 8:
                break

            digits = ""

    if len(digits) >= 8:

        try:

            t = datetime.strptime(
                digits[:8],
                "%Y%m%d"
            )

            return t.date()

        except Exception:

            pass

    return None


# ---------------------------------------------------------

def get_capture_time(image_path):

    """
    Priority

    EXIF

    ↓

    Filename

    ↓

    Unknown
    """

    t = get_exif_time(image_path)

    if t is not None:
        return t

    return get_filename_time(image_path)


# =========================================================
# Build image database
# =========================================================

image_infos = []

print("=" * 60)
print("Reading Dataset...")
print("=" * 60)

label_files = sorted(LABEL_DIR.glob("*.txt"))

for label_file in label_files:

    image_path = find_image(label_file.name)

    if image_path is None:
        continue

    classes, object_num = read_label(label_file)

    capture_time = get_capture_time(image_path)

    info = {

        "image": image_path,

        "label": label_file,

        "classes": classes,

        "main_class": Counter(classes).most_common(1)[0][0],

        "object_num": object_num,

        "date": capture_time,

    }

    image_infos.append(info)

print(f"Images Loaded : {len(image_infos)}")

print("=" * 60)
# =========================================================
# Part 2
# Group images by class and capture date
# =========================================================

print("Building class database...")

# ---------------------------------------------------------
# Group by main class
# ---------------------------------------------------------

class_database = defaultdict(list)

for info in image_infos:

    class_database[info["main_class"]].append(info)

print(f"Number of classes : {len(class_database)}")

# ---------------------------------------------------------
# Group by date inside each class
# ---------------------------------------------------------

class_date_database = {}

for cls_id, infos in class_database.items():

    date_groups = defaultdict(list)

    unknown_date = []

    for item in infos:

        if item["date"] is None:

            unknown_date.append(item)

        else:

            date_groups[item["date"]].append(item)

    # sort by date
    sorted_dates = sorted(date_groups.keys())

    ordered_groups = []

    for d in sorted_dates:

        ordered_groups.append({

            "date": d,

            "images": date_groups[d],

            "count": len(date_groups[d])

        })

    # append unknown date at the end
    if len(unknown_date) > 0:

        random.shuffle(unknown_date)

        ordered_groups.append({

            "date": "Unknown",

            "images": unknown_date,

            "count": len(unknown_date)

        })

    class_date_database[cls_id] = ordered_groups

# ---------------------------------------------------------
# Print date distribution
# ---------------------------------------------------------

print("=" * 60)
print("Capture Date Distribution")
print("=" * 60)

for cls in sorted(class_date_database.keys()):

    total = sum(x["count"] for x in class_date_database[cls])

    print(f"\nClass {cls}")

    print(f"Total Images : {total}")

    for group in class_date_database[cls]:

        print(f"   {group['date']} : {group['count']}")

# ---------------------------------------------------------
# Build class statistics
# ---------------------------------------------------------

class_statistics = {}

for cls, infos in class_database.items():

    image_num = len(infos)

    object_num = 0

    day_num = 0

    days = set()

    for item in infos:

        object_num += item["object_num"]

        if item["date"] is not None:

            days.add(item["date"])

    day_num = len(days)

    class_statistics[cls] = {

        "images": image_num,

        "objects": object_num,

        "days": day_num

    }

print("=" * 60)
print("Class Statistics")
print("=" * 60)

for cls in sorted(class_statistics.keys()):

    stat = class_statistics[cls]

    print(
        f"Class {cls:2d} | "
        f"Images: {stat['images']:5d} | "
        f"Objects: {stat['objects']:6d} | "
        f"Days: {stat['days']:3d}"
    )

# ---------------------------------------------------------
# Prepare split containers
# ---------------------------------------------------------

train_set = []
val_set = []
test_set = []

split_statistics = {

    "train": Counter(),

    "val": Counter(),

    "test": Counter()

}

print("=" * 60)
print("Database construction completed.")
print("=" * 60)
# =========================================================
# Part 3
# Split dataset by capture date
# =========================================================

print("=" * 60)
print("Splitting Dataset...")
print("=" * 60)

for cls in sorted(class_date_database.keys()):

    groups = class_date_database[cls]

    # -----------------------------------------------------
    # Separate dated and unknown groups
    # -----------------------------------------------------

    dated_groups = []
    unknown_images = []

    for group in groups:

        if group["date"] == "Unknown":

            unknown_images.extend(group["images"])

        else:

            dated_groups.append(group)

    total_images = sum(g["count"] for g in dated_groups)
    total_images += len(unknown_images)

    train_target = int(total_images * TRAIN_RATIO)
    val_target = int(total_images * VAL_RATIO)

    train_count = 0
    val_count = 0

    print(f"\nClass {cls}")
    print(f"Target: train={train_target}, val={val_target}, test={total_images-train_target-val_target}")

    # -----------------------------------------------------
    # Allocate dated images (whole day together)
    # -----------------------------------------------------

    for group in dated_groups:

        imgs = group["images"]

        if train_count < train_target:

            train_set.extend(imgs)

            train_count += len(imgs)

            split_statistics["train"][cls] += len(imgs)

        elif val_count < val_target:

            val_set.extend(imgs)

            val_count += len(imgs)

            split_statistics["val"][cls] += len(imgs)

        else:

            test_set.extend(imgs)

            split_statistics["test"][cls] += len(imgs)

    # -----------------------------------------------------
    # Allocate unknown-date images randomly
    # -----------------------------------------------------

    if len(unknown_images):

        random.shuffle(unknown_images)

        remain_train = max(0, train_target - train_count)
        remain_val = max(0, val_target - val_count)

        train_unknown = unknown_images[:remain_train]

        val_unknown = unknown_images[
            remain_train:
            remain_train + remain_val
        ]

        test_unknown = unknown_images[
            remain_train + remain_val:
        ]

        train_set.extend(train_unknown)
        val_set.extend(val_unknown)
        test_set.extend(test_unknown)

        split_statistics["train"][cls] += len(train_unknown)
        split_statistics["val"][cls] += len(val_unknown)
        split_statistics["test"][cls] += len(test_unknown)

# ---------------------------------------------------------
# Remove duplicates (safety check)
# ---------------------------------------------------------

def remove_duplicates(dataset):

    unique = {}
    result = []

    for item in dataset:

        key = item["image"].name

        if key not in unique:

            unique[key] = True
            result.append(item)

    return result

train_set = remove_duplicates(train_set)
val_set = remove_duplicates(val_set)
test_set = remove_duplicates(test_set)

# ---------------------------------------------------------
# Global duplicate check
# ---------------------------------------------------------

train_names = set(x["image"].name for x in train_set)
val_names = set(x["image"].name for x in val_set)
test_names = set(x["image"].name for x in test_set)

assert len(train_names & val_names) == 0
assert len(train_names & test_names) == 0
assert len(val_names & test_names) == 0

print("=" * 60)
print("Split Finished")
print("=" * 60)

print(f"Train : {len(train_set)}")
print(f"Val   : {len(val_set)}")
print(f"Test  : {len(test_set)}")

print()

print("Class Distribution")

all_classes = sorted(class_database.keys())

for cls in all_classes:

    tr = split_statistics["train"][cls]
    va = split_statistics["val"][cls]
    te = split_statistics["test"][cls]

    total = tr + va + te

    print(
        f"Class {cls:2d} | "
        f"Train {tr:5d} | "
        f"Val {va:5d} | "
        f"Test {te:5d} | "
        f"Total {total:5d}"
    )

print("=" * 60)
# =========================================================
# Part 4
# Copy images and labels
# =========================================================

print("=" * 60)
print("Creating output directories...")
print("=" * 60)

create_output_dirs()

# ---------------------------------------------------------
# Copy one subset
# ---------------------------------------------------------

def copy_subset(dataset, subset_name):
    """
    Copy one subset (train / val / test).
    """

    image_dst = OUTPUT_DIR / "images" / subset_name
    label_dst = OUTPUT_DIR / "labels" / subset_name

    total = len(dataset)

    print(f"\n{subset_name.upper()} ({total} images)")

    copied = 0

    for item in dataset:

        src_img = item["image"]
        src_label = item["label"]

        dst_img = image_dst / src_img.name
        dst_label = label_dst / src_label.name

        shutil.copy2(src_img, dst_img)
        shutil.copy2(src_label, dst_label)

        copied += 1

        # 每500张显示一次进度
        if copied % 500 == 0 or copied == total:

            percent = copied / total * 100

            print(
                f"\rCopied {copied:6d}/{total:<6d}"
                f" ({percent:5.1f}%)",
                end=""
            )

    print()

# ---------------------------------------------------------
# Start copying
# ---------------------------------------------------------

print("\nCopying training set...")
copy_subset(train_set, "train")

print("\nCopying validation set...")
copy_subset(val_set, "val")

print("\nCopying test set...")
copy_subset(test_set, "test")

print("\nAll files copied successfully.")

# ---------------------------------------------------------
# Verify copied files
# ---------------------------------------------------------

def count_files(folder, suffixes):

    num = 0

    for file in folder.iterdir():

        if file.suffix.lower() in suffixes:

            num += 1

    return num

print("\nVerifying dataset...")

train_imgs = count_files(
    OUTPUT_DIR / "images" / "train",
    IMAGE_SUFFIX
)

val_imgs = count_files(
    OUTPUT_DIR / "images" / "val",
    IMAGE_SUFFIX
)

test_imgs = count_files(
    OUTPUT_DIR / "images" / "test",
    IMAGE_SUFFIX
)

train_labels = count_files(
    OUTPUT_DIR / "labels" / "train",
    [".txt"]
)

val_labels = count_files(
    OUTPUT_DIR / "labels" / "val",
    [".txt"]
)

test_labels = count_files(
    OUTPUT_DIR / "labels" / "test",
    [".txt"]
)

print("=" * 60)
print("Verification")
print("=" * 60)

print(f"Train Images : {train_imgs}")
print(f"Train Labels : {train_labels}")

print()

print(f"Val Images   : {val_imgs}")
print(f"Val Labels   : {val_labels}")

print()

print(f"Test Images  : {test_imgs}")
print(f"Test Labels  : {test_labels}")

assert train_imgs == train_labels, "Train image/label mismatch!"
assert val_imgs == val_labels, "Val image/label mismatch!"
assert test_imgs == test_labels, "Test image/label mismatch!"

print("\nVerification passed.")

print("=" * 60)
# =========================================================
# Part 5
# Dataset statistics and log generation
# =========================================================

print("=" * 60)
print("Generating Statistics...")
print("=" * 60)

# ---------------------------------------------------------
# Statistics Function
# ---------------------------------------------------------

def dataset_statistics(dataset):

    image_num = len(dataset)

    object_num = 0

    class_counter = Counter()

    date_counter = Counter()

    for item in dataset:

        object_num += item["object_num"]

        for cls in item["classes"]:

            class_counter[cls] += 1

        if item["date"] is not None:

            date_counter[str(item["date"])] += 1

    return {

        "images": image_num,

        "objects": object_num,

        "classes": class_counter,

        "dates": date_counter

    }


train_stat = dataset_statistics(train_set)
val_stat = dataset_statistics(val_set)
test_stat = dataset_statistics(test_set)

# ---------------------------------------------------------
# Print Summary
# ---------------------------------------------------------

print("\nOverall Summary")
print("-" * 60)

total_images = (
    train_stat["images"] +
    val_stat["images"] +
    test_stat["images"]
)

total_objects = (
    train_stat["objects"] +
    val_stat["objects"] +
    test_stat["objects"]
)

print(f"Total Images : {total_images}")
print(f"Total Objects: {total_objects}")

print()

print(
    f"Train : {train_stat['images']:6d} "
    f"({train_stat['images']/total_images*100:.2f}%)"
)

print(
    f"Val   : {val_stat['images']:6d} "
    f"({val_stat['images']/total_images*100:.2f}%)"
)

print(
    f"Test  : {test_stat['images']:6d} "
    f"({test_stat['images']/total_images*100:.2f}%)"
)

# ---------------------------------------------------------
# Class Statistics
# ---------------------------------------------------------

print("\nClass Statistics")
print("-" * 60)

all_classes = sorted(
    set(train_stat["classes"].keys()) |
    set(val_stat["classes"].keys()) |
    set(test_stat["classes"].keys())
)

for cls in all_classes:

    tr = train_stat["classes"][cls]
    va = val_stat["classes"][cls]
    te = test_stat["classes"][cls]

    total = tr + va + te

    print(
        f"Class {cls:2d} | "
        f"Train {tr:6d} | "
        f"Val {va:6d} | "
        f"Test {te:6d} | "
        f"Total {total:6d}"
    )

# ---------------------------------------------------------
# Date Statistics
# ---------------------------------------------------------

print("\nCapture Date Statistics")
print("-" * 60)

all_dates = sorted(
    set(train_stat["dates"].keys()) |
    set(val_stat["dates"].keys()) |
    set(test_stat["dates"].keys())
)

for d in all_dates:

    print(

        f"{d} "

        f"T:{train_stat['dates'][d]:4d} "

        f"V:{val_stat['dates'][d]:4d} "

        f"E:{test_stat['dates'][d]:4d}"

    )

# ---------------------------------------------------------
# Save Log
# ---------------------------------------------------------

log_file = OUTPUT_DIR / "split_log.txt"

with open(log_file, "w", encoding="utf-8") as f:

    f.write("=" * 60 + "\n")
    f.write("KFP-YOLO Dataset Split Log\n")
    f.write("=" * 60 + "\n\n")

    f.write(f"Total Images : {total_images}\n")
    f.write(f"Total Objects: {total_objects}\n\n")

    f.write("Subset Statistics\n")
    f.write("------------------------------\n")

    f.write(
        f"Train : {train_stat['images']} images\n"
    )

    f.write(
        f"Val   : {val_stat['images']} images\n"
    )

    f.write(
        f"Test  : {test_stat['images']} images\n\n"
    )

    f.write("Class Statistics\n")
    f.write("------------------------------\n")

    for cls in all_classes:

        tr = train_stat["classes"][cls]
        va = val_stat["classes"][cls]
        te = test_stat["classes"][cls]

        total = tr + va + te

        f.write(

            f"Class {cls} : "

            f"{tr} "

            f"{va} "

            f"{te} "

            f"{total}\n"

        )

    f.write("\nCapture Dates\n")
    f.write("------------------------------\n")

    for d in all_dates:

        f.write(

            f"{d} "

            f"{train_stat['dates'][d]} "

            f"{val_stat['dates'][d]} "

            f"{test_stat['dates'][d]}\n"

        )

print("\nLog saved to:")

print(log_file)

# ---------------------------------------------------------
# Finished
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("Dataset Split Completed Successfully.")
print("=" * 60)

print(f"Output Directory : {OUTPUT_DIR.resolve()}")

print(f"Train Images : {len(train_set)}")
print(f"Val Images   : {len(val_set)}")
print(f"Test Images  : {len(test_set)}")

print("=" * 60)