import os
import cv2
import random

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

images_path = "dataset/LUNA16/images/train"
labels_path = "dataset/LUNA16/labels/train"

save_cancer = "classifier_dataset/cancer"
save_non = "classifier_dataset/non_cancer"

os.makedirs(save_cancer, exist_ok=True)
os.makedirs(save_non, exist_ok=True)

count = 0

for label_file in os.listdir(labels_path):

    if not label_file.endswith(".txt"):
        continue

    img_name = label_file.replace(".txt", ".jpg")
    img_path = os.path.join(images_path, img_name)
    label_path = os.path.join(labels_path, label_file)

    if not os.path.exists(img_path):
        print(f"Warning: Image not found {img_path}")
        continue

    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    
    h, w, _ = img.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls, x, y, bw, bh = map(float, parts)

        # YOLO format → pixel coords
        x1 = int((x - bw/2) * w)
        y1 = int((y - bh/2) * h)
        x2 = int((x + bw/2) * w)
        y2 = int((y + bh/2) * h)

        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0:
            print(f"Warning: Empty crop at {img_path}")
            continue

        save_path = os.path.join(save_cancer, f"cancer_{count}.jpg")
        cv2.imwrite(save_path, crop)
        count += 1

print(f"✓ Cancer dataset created! Total nodules extracted: {count}")

# -----------------------------
# Create NON-CANCER samples
# Random lung patches (negative sampling)
# -----------------------------

count_non = 0

for img_file in os.listdir(images_path):

    img_path = os.path.join(images_path, img_file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    h, w, _ = img.shape

    # Create 3 random crops from each image
    for i in range(3):
        x = random.randint(0, max(1, w-100))
        y = random.randint(0, max(1, h-100))

        crop = img[y:y+100, x:x+100]
        
        if crop.size == 0:
            continue

        save_path = os.path.join(save_non, f"non_{count_non}.jpg")
        cv2.imwrite(save_path, crop)
        count_non += 1

print(f"✓ Non-cancer dataset created! Total background patches: {count_non}")
print(f"\n📊 Final classifier dataset:")
print(f"   Cancer samples: {count}")
print(f"   Non-cancer samples: {count_non}")
print(f"   Total: {count + count_non}")
