import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# Set working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Dataset transforms
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Check if dataset exists
if not os.path.exists("classifier_dataset"):
    print("Error: classifier_dataset folder not found!")
    print("Please run build_classifier_dataset.py first")
    exit(1)

dataset = datasets.ImageFolder("classifier_dataset", transform=transform)

print(f"Total samples: {len(dataset)}")
print(f"Classes: {dataset.classes}")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

# ----------------------------
# Model (Transfer Learning)
# ----------------------------
print("\nLoading ResNet50...")
model = models.resnet50(pretrained=True)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for binary classification
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 2)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# ----------------------------
# Training loop
# ----------------------------
EPOCHS = 5

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    batch_count = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    avg_loss = total_loss / batch_count
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    avg_val_loss = val_loss / len(val_loader)
    print(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%\n")

# Save trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet_cancer.pth")

print("✓ Training complete!")
print("✓ Model saved to: models/resnet_cancer.pth")
