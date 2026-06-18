from ultralytics import YOLO
import cv2
from PIL import Image
from utils import risk_level
from preprocessing import preprocess_ct_scan
import os
import glob

# 🔥 NEW IMPORTS (Classifier)
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

# --------------------------------------------------
# 📂 PATH SETUP
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))

# ---------------- YOLO MODEL (Detection) ----------------
yolo_path = os.path.join(script_dir, "models/best.pt")
yolo_model = YOLO(yolo_path)

# ---------------- RESNET MODEL (Classification) ----------------
device = torch.device("cpu")

classifier = models.resnet50(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, 2)

resnet_path = os.path.join(script_dir, "models/resnet_cancer.pth")
classifier.load_state_dict(torch.load(resnet_path, map_location=device))
classifier.eval()

# Image transform for classifier
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# --------------------------------------------------
# 🔍 FIND YOLO OUTPUT IMAGE
# --------------------------------------------------
def get_detected_image_path():
    predict_folders = glob.glob("runs/detect/predict*")
    if not predict_folders:
        return None

    latest_folder = max(predict_folders, key=os.path.getctime)
    detected_images = glob.glob(f"{latest_folder}/*.jpg")

    if len(detected_images) > 0:
        return detected_images[0]

    return None


# --------------------------------------------------
# 🚀 MAIN PIPELINE
# --------------------------------------------------
def run_pipeline(image_path):

    # Step 1️⃣ Preprocess CT scan
    processed_img = preprocess_ct_scan(image_path)

    temp_path = "temp_processed.jpg"
    cv2.imwrite(temp_path, processed_img)

    # Step 2️⃣ YOLO Detection
    results = yolo_model(temp_path, save=True,
                         project="runs",
                         name="detect",
                         exist_ok=True)

    boxes = results[0].boxes.xyxy
    scores = results[0].boxes.conf

    diameters = []
    volumes = []
    confidences = []
    slice_thickness = 1

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        width = x2 - x1
        height = y2 - y1

        diameter = max(width, height)
        area = width * height
        volume = area * slice_thickness

        diameters.append(diameter)
        volumes.append(volume)
        confidences.append(float(scores[i]))

    # --------------------------------------------------
    # 🧠 REAL AI CLASSIFICATION USING RESNET
    # --------------------------------------------------
    if len(boxes) == 0:
        prediction = "Non-Cancer"
        confidence_cls = 0.99

    else:
        # Crop first detected tumour
        x1, y1, x2, y2 = map(int, boxes[0])
        crop = processed_img[y1:y2, x1:x2]

        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        crop_pil = Image.fromarray(crop)

        input_tensor = transform(crop_pil).unsqueeze(0)

        with torch.no_grad():
            outputs = classifier(input_tensor)
            probs = torch.softmax(outputs, dim=1)

        cancer_prob = float(probs[0][1])
        confidence_cls = cancer_prob

        if cancer_prob > 0.5:
            prediction = "Cancer"
        else:
            prediction = "Non-Cancer"

    # --------------------------------------------------
    # 📊 RISK ANALYSIS
    # --------------------------------------------------
    risk = risk_level(len(boxes), diameters, prediction, confidences)

    # --------------------------------------------------
    # 🔥 HEATMAP + OVERLAY
    # --------------------------------------------------
    img_vis = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
    heatmap = cv2.applyColorMap(processed_img, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)

    cv2.imwrite("heatmap.jpg", heatmap)
    cv2.imwrite("overlay.jpg", overlay)

    # --------------------------------------------------
    # 📷 DETECTED IMAGE PATH
    # --------------------------------------------------
    detected_img_path = get_detected_image_path()
    detected_img = None
    if detected_img_path and os.path.exists(detected_img_path):
        detected_img = cv2.imread(detected_img_path)

    # --------------------------------------------------
    # 📦 FINAL OUTPUT
    # --------------------------------------------------
    return {
        "nodules": len(boxes),
        "max_diameter": max(diameters) if diameters else 0,
        "avg_volume": sum(volumes)/len(volumes) if volumes else 0,
        "prediction": prediction,
        "confidence": confidence_cls,
        "risk": risk,
        "image": image_path,
        "detected_image": detected_img,
        "detected_image_path": detected_img_path,
        "heatmap": heatmap,
        "overlay": overlay
    }