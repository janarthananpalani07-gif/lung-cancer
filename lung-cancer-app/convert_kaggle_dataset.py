import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE = "lung_colon_image_set/lung_image_sets"

CANCER_FOLDERS = ["lung_aca", "lung_scc"]
NORMAL_FOLDER  = "lung_n"

DEST = "dataset"

# create folders
for split in ["train","val","test"]:
    os.makedirs(f"{DEST}/{split}/cancer", exist_ok=True)
    os.makedirs(f"{DEST}/{split}/non_cancer", exist_ok=True)

def split_and_copy(images, label):
    train, temp = train_test_split(images, test_size=0.3, random_state=42)
    val, test   = train_test_split(temp, test_size=0.5, random_state=42)

    for img in train:
        shutil.copy(img, f"{DEST}/train/{label}")
    for img in val:
        shutil.copy(img, f"{DEST}/val/{label}")
    for img in test:
        shutil.copy(img, f"{DEST}/test/{label}")

# collect cancer images
cancer_images = []
for folder in CANCER_FOLDERS:
    path = os.path.join(SOURCE, folder)
    for img in os.listdir(path):
        cancer_images.append(os.path.join(path, img))

# collect normal images
normal_images = []
path = os.path.join(SOURCE, NORMAL_FOLDER)
for img in os.listdir(path):
    normal_images.append(os.path.join(path, img))

# split and copy
split_and_copy(cancer_images, "cancer")
split_and_copy(normal_images, "non_cancer")

print("✅ Kaggle dataset converted successfully!")