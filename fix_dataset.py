import os, shutil, random

base = "dataset"

train_cancer = os.path.join(base, "train/cancer")
train_non = os.path.join(base, "train/non_cancer")

val_cancer = os.path.join(base, "val/cancer")
val_non = os.path.join(base, "val/non_cancer")

os.makedirs(val_cancer, exist_ok=True)
os.makedirs(val_non, exist_ok=True)

def move_20_percent(src, dst):
    files = os.listdir(src)
    random.shuffle(files)
    split = int(len(files) * 0.2)
    val_files = files[:split]

    for f in val_files:
        shutil.move(os.path.join(src, f), os.path.join(dst, f))

move_20_percent(train_cancer, val_cancer)
move_20_percent(train_non, val_non)

print("✅ Validation dataset created (80/20 split)")