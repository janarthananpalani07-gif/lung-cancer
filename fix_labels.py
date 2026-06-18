import os
import shutil

SOURCE = "dataset/LUNA16"
DEST   = "dataset"

splits = ["train","valid","test"]

for split in splits:

    src_split = os.path.join(SOURCE, split)

    cancer_src = os.path.join(src_split, "cancer")
    non_src1   = os.path.join(src_split, "non cancer")
    non_src2   = os.path.join(src_split, "normal")

    dest_cancer = os.path.join(DEST, split, "cancer")
    dest_non    = os.path.join(DEST, split, "non_cancer")

    os.makedirs(dest_cancer, exist_ok=True)
    os.makedirs(dest_non, exist_ok=True)

    # copy cancer images
    if os.path.exists(cancer_src):
        for img in os.listdir(cancer_src):
            shutil.copy(os.path.join(cancer_src, img), dest_cancer)

    # copy non cancer images
    for folder in [non_src1, non_src2]:
        if os.path.exists(folder):
            for img in os.listdir(folder):
                shutil.copy(os.path.join(folder, img), dest_non)

print("✅ DATASET FIXED SUCCESSFULLY")