"""
Data loading and preprocessing for CIFAR-10.
"""

import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import BATCH_SIZE, DATA_DIR, DEVICE, NUM_WORKERS, PREFETCH

warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    module=r"torchvision\.datasets\.cifar",
)


def get_transforms():
    """Return train and test transforms for CIFAR-10."""
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    return train_tf, test_tf


def get_datasets(seed, forget_ratio=0.10):
    """
    Load CIFAR-10 and split into forget, retain, and test sets.
    
    Args:
        seed: Random seed for reproducibility.
        forget_ratio: Fraction of training data to forget.
        
    Returns:
        Tuple of (full_train, forget_ds, retain_ds, test_ds)
    """
    train_tf, test_tf = get_transforms()
    full_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    rng = np.random.default_rng(seed)
    n = len(full_train)
    forget_idx = rng.choice(n, size=int(forget_ratio * n), replace=False)
    retain_idx = np.setdiff1d(np.arange(n), forget_idx)

    forget_ds = Subset(full_train, forget_idx.tolist())
    retain_ds = Subset(full_train, retain_idx.tolist())
    return full_train, forget_ds, retain_ds, test_ds


def make_loader(dataset, shuffle=True, batch_size=BATCH_SIZE):
    """
    Create a DataLoader for a dataset.
    
    Args:
        dataset: PyTorch dataset.
        shuffle: Whether to shuffle the data.
        batch_size: Batch size for the loader.
        
    Returns:
        DataLoader instance.
    """
    pin_memory = DEVICE.type == "cuda"
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )
    if NUM_WORKERS > 0:
        kwargs["prefetch_factor"] = PREFETCH
    return DataLoader(dataset, **kwargs)
