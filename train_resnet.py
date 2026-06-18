import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# PATHS  (Data/test used as validation — no valid/ split exists)
# -----------------------------
train_dir = "LUNG CANCER/train"
val_dir   = "LUNG CANCER/test"

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# LOAD DATASET
# -----------------------------
train_dataset = ImageFolder(train_dir, transform=transform)
val_dataset   = ImageFolder(val_dir,   transform=val_transform)

print("Class mapping:", train_dataset.class_to_idx)
print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# 'Normal' = non-cancer, 'Lung Cancer' = cancer
NORMAL_CLASS = 'Normal'
normal_idx = train_dataset.class_to_idx[NORMAL_CLASS]

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False, num_workers=0)

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
    Normal      -> 0  (non-cancer / NON_CANCER_CLASS_INDEX)
    Lung Cancer -> 1  (cancer / CANCER_CLASS_INDEX)
    """
    return torch.tensor([0 if lbl == normal_idx else 1 for lbl in labels])

# -----------------------------
# TRAIN LOOP
# -----------------------------
epochs = 10

for epoch in range(epochs):
    model.train()
    correct = total = 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs   = imgs.to(device)
        labels = convert_to_binary(labels).to(device)

        outputs = model(imgs)
        loss    = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} | Loss {loss.item():.4f}")

    train_acc = correct / total

    # VALIDATION
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs   = imgs.to(device)
            labels = convert_to_binary(labels).to(device)

            outputs     = model(imgs)
            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

    val_acc = correct / total
    print(f"Epoch {epoch+1}/{epochs} | Train Acc {train_acc:.3f} | Val Acc {val_acc:.3f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)
save_path = "models/resnet_ct_cancer.pth"
torch.save(model.state_dict(), save_path)
print(f"✅ CT ResNet Training Finished! Model saved to {save_path}")