import os
import shutil

src = "Data"
dst = "ct_dataset"

splits = ["train", "valid", "test"]

for split in splits:
    os.makedirs(f"{dst}/{split}/cancer", exist_ok=True)
    os.makedirs(f"{dst}/{split}/non_cancer", exist_ok=True)

    split_path = os.path.join(src, split)

    for folder in os.listdir(split_path):
        folder_path = os.path.join(split_path, folder)

        # NORMAL → non cancer
        if "normal" in folder.lower():
            target = f"{dst}/{split}/non_cancer"
        else:
            target = f"{dst}/{split}/cancer"

        for img in os.listdir(folder_path):
            src_img = os.path.join(folder_path, img)
            dst_img = os.path.join(target, img)
            shutil.copy(src_img, dst_img)

print("✅ CT dataset fixed to 2 classes!")