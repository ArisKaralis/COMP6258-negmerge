"""
Model building and management for ResNet-18 on CIFAR-10.
"""

import copy

import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE


def build_resnet18(num_classes=10):
    """
    Build and return a ResNet-18 model for CIFAR-10.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10).
        
    Returns:
        ResNet-18 model on the designated device.
    """
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(DEVICE)


def clone_model(model):
    """Create a deep copy of a model."""
    return copy.deepcopy(model)
