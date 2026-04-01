"""
Shared utilities for the CIFAR-10 classifier unlearning experiments.

Contains: dataset loading, model definitions, task vectors, and evaluation metrics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
NUM_CLASSES  = 10


def cifar10_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    return train_tf, eval_tf


def make_forget_retain_indices(n_total, forget_fraction=0.1, seed=42):
    """Returns (forget_indices, retain_indices) as numpy int arrays."""
    rng = np.random.default_rng(seed)
    n_forget = int(n_total * forget_fraction)
    forget_idx = rng.choice(n_total, size=n_forget, replace=False)
    retain_idx = np.setdiff1d(np.arange(n_total), forget_idx)
    return forget_idx, retain_idx


def get_cifar10_loaders(
    data_dir: str,
    forget_fraction: float = 0.1,
    seed: int = 42,
    batch_size: int = 128,
    num_workers: int = 4,
    forget_indices=None,  # pass pre-computed to reuse the same split
    retain_indices=None,
):
    """
    Returns a dict with DataLoaders and metadata:
      full_train, forget_train, retain_train   - augmented (for training)
      forget_eval, retain_eval, test           - no augmentation (for evaluation)
      forget_indices, retain_indices           - numpy arrays
      num_classes
    """
    train_tf, eval_tf = cifar10_transforms()

    full_train_aug  = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True,  transform=train_tf)
    full_train_eval = torchvision.datasets.CIFAR10(data_dir, train=True,  download=False, transform=eval_tf)
    test_dataset    = torchvision.datasets.CIFAR10(data_dir, train=False, download=True,  transform=eval_tf)

    if forget_indices is None or retain_indices is None:
        forget_indices, retain_indices = make_forget_retain_indices(
            len(full_train_aug), forget_fraction, seed
        )

    def loader(ds, shuffle=False):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True)

    return {
        'full_train':   loader(full_train_aug, shuffle=True),
        'forget_train': loader(Subset(full_train_aug,  forget_indices), shuffle=True),
        'retain_train': loader(Subset(full_train_aug,  retain_indices), shuffle=True),
        'forget_eval':  loader(Subset(full_train_eval, forget_indices)),
        'retain_eval':  loader(Subset(full_train_eval, retain_indices)),
        'test':         loader(test_dataset),
        'forget_indices': forget_indices,
        'retain_indices': retain_indices,
        'num_classes': NUM_CLASSES,
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def get_model(model_name: str, num_classes: int = 10) -> nn.Module:
    """
    Returns an untrained model suitable for CIFAR-10 (32x32 images).

    Supported: resnet18, vgg16, swin_t
    """
    name = model_name.lower().replace('-', '_')

    if name == 'resnet18':
        model = tv_models.resnet18(weights=None)
        # Replace the 7x7/stride-2 stem with a 3x3/stride-1 conv for 32x32 inputs
        model.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        model.fc      = nn.Linear(model.fc.in_features, num_classes)

    elif name == 'vgg16':
        model = tv_models.vgg16(weights=None)
        # Replace the final classifier block to match CIFAR-10 feature dimensions
        model.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    elif name in ('swin_t', 'swint'):
        try:
            import timm
            model = timm.create_model(
                'swin_tiny_patch4_window7_224',
                pretrained=False,
                num_classes=num_classes,
                img_size=32,
            )
        except ImportError:
            raise ImportError("timm is required for Swin-T. Install with: pip install timm")

    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: resnet18, vgg16, swin_t")

    return model


# ---------------------------------------------------------------------------
# Task Vectors
# ---------------------------------------------------------------------------

class ClassifierTaskVector:
    """
    Task vector for standard classifier models.

    vector = finetuned_state_dict - pretrained_state_dict
    All arithmetic is done in float32; result is cast back to original dtype on apply.
    """

    def __init__(self, pretrained_sd=None, finetuned_sd=None, vector=None):
        if vector is not None:
            self.vector = vector
            return
        assert pretrained_sd is not None and finetuned_sd is not None
        with torch.no_grad():
            self.vector = {}
            for key, p_val in pretrained_sd.items():
                if p_val.dtype in (torch.int64, torch.uint8, torch.bool):
                    continue
                if key not in finetuned_sd:
                    continue
                self.vector[key] = finetuned_sd[key].float() - p_val.float()

    def __neg__(self):
        return ClassifierTaskVector(vector={k: -v for k, v in self.vector.items()})

    def __mul__(self, scalar):
        return ClassifierTaskVector(vector={k: scalar * v for k, v in self.vector.items()})

    __rmul__ = __mul__

    def apply_to(self, pretrained_sd: dict, scaling_coef: float = 1.0) -> dict:
        """Returns a new state dict = pretrained + scaling_coef * vector."""
        with torch.no_grad():
            new_sd = {}
            for key, p_val in pretrained_sd.items():
                if key not in self.vector:
                    new_sd[key] = p_val.clone()
                else:
                    updated = p_val.float() + scaling_coef * self.vector[key]
                    new_sd[key] = updated.to(p_val.dtype)
        return new_sd


def negmerge(task_vectors: list) -> ClassifierTaskVector:
    """
    Merge a list of ClassifierTaskVectors using sign-consensus (NegMerge).

    Returns the NEGATED merged vector (ready for unlearning via apply_to).
    Sparsity info is printed to stdout.
    """
    n = len(task_vectors)
    assert n > 0

    # Accumulate sum and sign sum
    merged = {k: torch.zeros_like(v) for k, v in task_vectors[0].vector.items()}
    sign_sum = {k: torch.zeros_like(v) for k, v in task_vectors[0].vector.items()}

    for tv in task_vectors:
        for key in merged:
            merged[key] += tv.vector[key]
            sign_sum[key] += torch.sign(tv.vector[key])

    # Mask: keep only parameters unanimous in sign across all n models
    n_total, n_kept = 0, 0
    for key in merged:
        consensus = torch.abs(sign_sum[key]) == n
        merged[key] = torch.where(consensus, merged[key] / n, torch.zeros_like(merged[key]))
        n_total += consensus.numel()
        n_kept += int(consensus.sum().item())

    sparsity = 1.0 - n_kept / n_total
    print(f"NegMerge sparsity: {100 * sparsity:.1f}%  "
          f"({n_kept:,} / {n_total:,} parameters kept, {n} models merged)")

    return -ClassifierTaskVector(vector=merged)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device) -> float:
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return correct / total


@torch.no_grad()
def per_sample_losses(model: nn.Module, loader: DataLoader, device) -> np.ndarray:
    """Returns an array of per-sample cross-entropy losses."""
    model.eval()
    model.to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        out = model(images)
        losses.extend(criterion(out, labels).cpu().numpy())
    return np.array(losses)


def mia_auc(forget_losses: np.ndarray, test_losses: np.ndarray) -> float:
    """
    Membership Inference Attack AUC using a loss threshold.

    Classifies samples as 'members' (forget set) if they have low loss.
    AUC ~ 0.5 → model cannot distinguish forget from test set → good unlearning.
    AUC ~ 1.0 → forget set still memorised → bad unlearning.
    """
    # Labels: 1 = forget (member), 0 = test (non-member)
    labels = np.concatenate([np.ones(len(forget_losses)), np.zeros(len(test_losses))])
    # Membership score: low loss → high membership → negate for standard AUC convention
    scores = np.concatenate([-forget_losses, -test_losses])

    # Compute AUC manually (no sklearn dependency)
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return float(np.trapz(tpr, fpr))


def evaluate_unlearning(model: nn.Module, loaders: dict, device, label: str = '') -> dict:
    """
    Full unlearning evaluation: forget/retain/test accuracy + MIA AUC.

    Expected AUC for well-unlearned model: ~0.5 (indistinguishable from non-members).
    """
    tag = f"[{label}] " if label else ''
    results = {}

    results['forget_acc'] = accuracy(model, loaders['forget_eval'], device)
    results['retain_acc'] = accuracy(model, loaders['retain_eval'], device)
    results['test_acc']   = accuracy(model, loaders['test'], device)

    forget_losses = per_sample_losses(model, loaders['forget_eval'], device)
    test_losses   = per_sample_losses(model, loaders['test'], device)
    results['mia_auc'] = mia_auc(forget_losses, test_losses)

    print(f"\n{tag}Forget accuracy: {results['forget_acc']:.4f}  "
          f"(random chance = {1/NUM_CLASSES:.4f})")
    print(f"{tag}Retain accuracy: {results['retain_acc']:.4f}")
    print(f"{tag}Test accuracy:   {results['test_acc']:.4f}")
    print(f"{tag}MIA AUC:         {results['mia_auc']:.4f}  "
          f"(0.5 = perfect unlearning, 1.0 = still memorised)")

    return results
