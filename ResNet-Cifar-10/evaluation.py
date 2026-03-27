"""
Evaluation metrics including MIA-Efficacy.
"""

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, MIA_SHADOW_EPOCHS


def compute_losses(model, loader):
    """
    Compute per-sample cross-entropy losses.
    
    Args:
        model: PyTorch model.
        loader: DataLoader.
        
    Returns:
        NumPy array of per-sample losses.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses = []
    with torch.no_grad():
        for x, y in loader:
            non_blocking = DEVICE.type == "cuda"
            x = x.to(DEVICE, non_blocking=non_blocking)
            y = y.to(DEVICE, non_blocking=non_blocking)
            if DEVICE.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            loss = criterion(model(x), y)
            losses.extend(loss.cpu().numpy())
    return np.array(losses)


def mia_efficacy(unlearned_model, forget_loader, retain_loader,
                 shadow_epochs=MIA_SHADOW_EPOCHS):
    """
    Compute MIA-Efficacy using loss-threshold attack.
    
    Following Fan et al. 2024 / Carlini et al. 2022:
    - Label forget samples as "member" if their loss on the unlearned model is LOW.
    - MIA-Efficacy = fraction of forget samples correctly identified as members.
    - Higher score indicates the model still "remembers" forget data.
    
    Args:
        unlearned_model: The unlearned/unlearning model.
        forget_loader: DataLoader for forget set.
        retain_loader: DataLoader for retain set.
        shadow_epochs: (Unused, kept for compatibility).
        
    Returns:
        MIA-Efficacy score (0-100).
    """
    forget_losses = compute_losses(unlearned_model, forget_loader)
    retain_losses = compute_losses(unlearned_model, retain_loader)

    # Threshold: median retain loss (retain samples are "members")
    threshold = np.median(retain_losses)

    # A forget sample is predicted "member" if its loss < threshold
    predicted_member = forget_losses < threshold
    # MIA-Efficacy = fraction of forget samples predicted as member
    efficacy = 100.0 * predicted_member.mean()
    return efficacy


def compute_sparsity(task_vector):
    """
    Compute sparsity metrics for a task vector (dict of tensors).
    
    Args:
        task_vector: Dictionary mapping parameter names to tensors.
        
    Returns:
        Dictionary with sparsity metrics:
        - total_params: Total number of parameters.
        - nonzero_params: Number of non-zero parameters.
        - sparsity_pct: Percentage of parameters that are zero.
        - avg_magnitude: Average absolute value of non-zero params.
    """
    total = 0
    nonzero = 0
    sum_mag = 0.0

    for key, tensor in task_vector.items():
        flat = tensor.flatten()
        total += flat.numel()
        nz = (flat.abs() > 1e-10).sum().item()
        nonzero += nz
        sum_mag += flat.abs().sum().item()

    sparsity_pct = 100.0 * (total - nonzero) / total if total > 0 else 0
    avg_magnitude = (sum_mag / nonzero) if nonzero > 0 else 0

    return {
        'total_params': total,
        'nonzero_params': nonzero,
        'sparsity_pct': sparsity_pct,
        'avg_magnitude': avg_magnitude,
    }
