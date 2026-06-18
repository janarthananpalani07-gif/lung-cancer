import os
from pipeline import run_pipeline

cancer_path = "test_dataset/cancer"
normal_path = "test_dataset/non_cancer"

total = 0
correct = 0

tp = tn = fp = fn = 0

print("\n🔬 TESTING CANCER IMAGES\n")

# -------- TEST CANCER --------
for img in os.listdir(cancer_path):
    path = os.path.join(cancer_path, img)
    result = run_pipeline(path)

    pred = result["prediction"]
    conf = result["confidence"]

    print(f"{img} → {pred} ({conf:.2f})")

    total += 1

    if pred == "Cancer":
        correct += 1
        tp += 1
    else:
        fn += 1

# -------- TEST NORMAL --------
print("\n🔬 TESTING NORMAL IMAGES\n")

for img in os.listdir(normal_path):
    path = os.path.join(normal_path, img)
    result = run_pipeline(path)

    pred = result["prediction"]
    conf = result["confidence"]

    print(f"{img} → {pred} ({conf:.2f})")

    total += 1

    if pred == "Non-Cancer":
        correct += 1
        tn += 1
    else:
        fp += 1

# -------- METRICS --------
accuracy = correct / total
precision = tp / (tp + fp + 1e-6)
recall = tp / (tp + fn + 1e-6)
f1 = 2 * precision * recall / (precision + recall + 1e-6)

print("\n==============================")
print("RESULTS")
print("==============================")
print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")
print(f"F1 Score  : {f1:.2f}")