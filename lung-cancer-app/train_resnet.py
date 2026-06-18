import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# PATHS
# -----------------------------
train_dir = "Data/train"
val_dir   = "Data/valid"

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

# -----------------------------
# LOAD DATASET
# -----------------------------
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset   = ImageFolder(val_dir, transform=transform)

print("Class mapping:", train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)

# -----------------------------
# MODEL
# -----------------------------
model = models.resnet50(weights="IMAGENET1K_V1")
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# -----------------------------
# 🔥 LABEL CONVERTER FUNCTION
# -----------------------------
def convert_to_binary(labels):
    """
    normal -> 0 (non cancer)
    anything else -> 1 (cancer)
    """
    binary = []
    for lbl in labels:
        if lbl == train_dataset.class_to_idx['normal']:
            binary.append(0)
        else:
            binary.append(1)
    return torch.tensor(binary)

# -----------------------------
# TRAIN LOOP
# -----------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    correct = total = 0

    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = convert_to_binary(labels).to(device)   # ⭐ FIX HERE

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = correct/total

    # VALIDATION
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = convert_to_binary(labels).to(device)  # ⭐ FIX HERE

            outputs = model(imgs)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    val_acc = correct/total
    print(f"Epoch {epoch+1} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
torch.save(model.state_dict(), "models/resnet_ct_cancer.pth")
print("✅ CT ResNet Training Finished!")