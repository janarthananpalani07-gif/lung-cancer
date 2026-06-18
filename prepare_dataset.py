import os
import shutil
from glob import glob
import cv2
import numpy as np

# SOURCE PATHS (your current dataset)
IMAGE_ROOT = "dataset/LUNA16/images"
MASK_ROOT  = "dataset/LUNA16/masks"

# TARGET PATH (new dataset for ResNet)
TARGET_ROOT = "dataset"

splits = ["train", "valid", "test"]

def create_folders():
    for split in splits:
        os.makedirs(f"{TARGET_ROOT}/{split}/cancer", exist_ok=True)
        os.makedirs(f"{TARGET_ROOT}/{split}/normal", exist_ok=True)

def is_cancer(mask_path):
    mask = cv2.imread(mask_path, 0)
    return np.sum(mask) > 0   # if mask has white pixels → tumour exists

def process_split(split):
    image_paths = glob(f"{IMAGE_ROOT}/{split}/*.png")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        mask_path = f"{MASK_ROOT}/{split}/{filename}"

        if not os.path.exists(mask_path):
            continue

        if is_cancer(mask_path):
            dest = f"{TARGET_ROOT}/{split}/cancer/{filename}"
        else:
            dest = f"{TARGET_ROOT}/{split}/normal/{filename}"

        shutil.copy(img_path, dest)

create_folders()

for s in splits:
    process_split(s)

print("✅ Dataset prepared successfully!")