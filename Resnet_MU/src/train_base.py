import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import os

# 1. Setup Device (Optimized for your M2 Mac)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 2. Data Preparation (Standard normalization, no augmentation as per paper)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 3. Create the 10% Forget / 90% Retain Split 
indices = list(range(len(full_train_dataset)))
torch.manual_seed(42) 
indices = torch.randperm(len(full_train_dataset)).tolist()

forget_idx = indices[:5000]   # 10% Forget Set
retain_idx = indices[5000:]  # 90% Retain Set

# Save indices so 27 variants use the exact same data
os.makedirs("./metadata", exist_ok=True)
torch.save(forget_idx, "./metadata/forget_idx.pt")
torch.save(retain_idx, "./metadata/retain_idx.pt")

# 4. Define the Training Function for the "Original Model"
def train_original_model():
    # Load standard ResNet-18
    model = models.resnet18(num_classes=10).to(device)
    
    # Paper Settings: Batch 256, LR 0.05, SGD 
    train_loader = DataLoader(full_train_dataset, batch_size=256, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    print("Training Original Model (θ_ori) on 100% of CIFAR-10...")
    model.train()
    for epoch in range(100): # Standard training length for a base model
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch} complete.")

    torch.save(model.state_dict(), "resnet18_cifar10_original.pt")
    print("Original model saved as resnet18_cifar10_original.pt")

if __name__ == "__main__":
    train_original_model()