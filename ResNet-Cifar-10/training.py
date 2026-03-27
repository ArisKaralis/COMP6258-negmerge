"""
Training utilities for ResNet-18 on CIFAR-10.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from config import DEVICE, LR


def train_one_epoch(model, loader, optimizer, criterion):
    """Train the model for one epoch."""
    model.train()
    for x, y in loader:
        non_blocking = DEVICE.type == "cuda"
        x = x.to(DEVICE, non_blocking=non_blocking)
        y = y.to(DEVICE, non_blocking=non_blocking)
        if DEVICE.type == "cuda":
            x = x.contiguous(memory_format=torch.channels_last)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    """Return top-1 accuracy (%) on a dataloader."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            non_blocking = DEVICE.type == "cuda"
            x = x.to(DEVICE, non_blocking=non_blocking)
            y = y.to(DEVICE, non_blocking=non_blocking)
            if DEVICE.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total


def train_model(model, loader, epochs, lr=LR, weight_decay=1e-4,
                label_smoothing=0.0, desc="Training"):
    """
    Train model for a fixed number of epochs.
    
    Args:
        model: PyTorch model.
        loader: DataLoader for training.
        epochs: Number of training epochs.
        lr: Learning rate.
        weight_decay: Weight decay for optimizer.
        label_smoothing: Label smoothing for loss.
        desc: Description for progress bar.
        
    Returns:
        Trained model.
    """
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()
    return model
