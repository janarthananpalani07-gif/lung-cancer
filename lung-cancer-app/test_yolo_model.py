from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train-5/weights/best.pt")

image_path = "dataset/LUNA16/images/test/107939_JPG_jpg.rf.958ed467f4dce47823a044cefb8677ae.jpg"
img = cv2.imread(image_path)

results = model(image_path)
boxes = results[0].boxes.xyxy

if len(boxes) == 0:
    print("No tumour detected")
else:
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        crop = img[y1:y2, x1:x2]
        cv2.imwrite(f"tumour_crop_{i}.jpg", crop)

        print("Tumour cropped and saved!")