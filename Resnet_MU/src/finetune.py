import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms, datasets
import itertools
import os

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
os.makedirs("./model_pool", exist_ok=True)

# Load data and indices
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
forget_idx = torch.load("./metadata/forget_idx.pt")
forget_loader = DataLoader(Subset(full_train, forget_idx), batch_size=256, shuffle=True)

# Hyperparameter Grid
epochs_list = [40, 50, 60]
wd_list = [1e-4, 5e-5, 1e-5]
ls_list = [0, 0.05, 0.1]
grid = list(itertools.product(epochs_list, wd_list, ls_list))

def train_variant(config, variant_id):
    e, wd, ls = config
    
    model = models.resnet18(num_classes=10)
    model.load_state_dict(torch.load("resnet18_cifar10_original.pt"))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=ls)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=wd)

    print(f"Training Variant {variant_id}: Epochs={e}, WD={wd}, LS={ls}")
    model.train()
    for _ in range(e):
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Save model checkpoint
    torch.save(model.state_dict(), f"./checkpoints/variant_{variant_id}.pt")
    
    del model, optimizer
    torch.mps.empty_cache()

for i, config in enumerate(grid):
    train_variant(config, i + 1)