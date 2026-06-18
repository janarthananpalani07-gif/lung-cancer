from ultralytics import YOLO
import cv2
from pipeline import classify_tumor

model = YOLO("runs/detect/train-5/weights/best.pt")

def detect_tumor_yolo(image_path):
    img = cv2.imread(image_path)
    results = model(image_path)

    boxes = results[0].boxes.xyxy

    if len(boxes) == 0:
        return "No Tumor Detected"

    predictions = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        crop_path = f"crop_{i}.jpg"
        cv2.imwrite(crop_path, crop)

        # 🔥 Send crop to ResNet
        result = classify_tumor(crop_path)
        predictions.append(result)

    return predictions