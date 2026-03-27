"""
Performance optimization utilities for faster evaluation and search.
"""

import numpy as np
import torch
from tqdm import tqdm

from config import DEVICE


def batch_evaluate(model, loaders_dict):
    """
    Evaluate model on multiple loaders in one pass.
    
    Args:
        model: PyTorch model.
        loaders_dict: Dict of {name: loader} pairs.
        
    Returns:
        Dict of {name: accuracy}.
    """
    results = {}
    model.eval()
    with torch.no_grad():
        for name, loader in loaders_dict.items():
            correct = total = 0
            for x, y in loader:
                non_blocking = DEVICE.type == "cuda"
                x = x.to(DEVICE, non_blocking=non_blocking)
                y = y.to(DEVICE, non_blocking=non_blocking)
                if DEVICE.type == "cuda":
                    x = x.contiguous(memory_format=torch.channels_last)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
            results[name] = 100.0 * correct / total
    return results


def adaptive_coef_range(coef_range, initial_best, tolerance=2.0):
    """
    Adapt coefficient search range based on initial search.
    
    Focuses search around promising coefficients to reduce iterations.
    
    Args:
        coef_range: Original coefficient range [c1, c2, ..., cn].
        initial_best: Best coefficient found so far.
        tolerance: Search within ±tolerance around initial_best.
        
    Returns:
        Refined coefficient range.
    """
    if initial_best is None or tolerance is None:
        return coef_range
    
    refined = [c for c in coef_range if abs(c - initial_best) <= tolerance]
    # Always include original best
    if initial_best not in refined:
        refined.append(initial_best)
    return sorted(set(refined))


def early_stop_coef_search(model_evals, window_size=3):
    """
    Detect if coefficient search is plateauing (early stopping).
    
    Args:
        model_evals: List of forget accuracies in order tested.
        window_size: Number of evaluations to check for improvement.
        
    Returns:
        True if improvement plateaus, False otherwise.
    """
    if len(model_evals) < window_size:
        return False
    
    recent = np.array(model_evals[-window_size:])
    # Stop if no improvement in last window_size evals
    return np.max(recent[:-1]) >= recent[-1]


def sparsity_aware_coef_range(task_vector, base_range=None, sparsity_threshold=0.5):
    """
    Suggest coefficient range based on task vector sparsity.
    
    Sparse task vectors (high sign consensus) may need different range.
    
    Args:
        task_vector: Task vector dict.
        base_range: Default range to use if no adaptation needed.
        sparsity_threshold: Sparsity % to trigger adaptation.
        
    Returns:
        Suggested coefficient range.
    """
    if base_range is None:
        base_range = [i * 0.05 for i in range(1, 21)]
    
    # Compute sparsity
    total = 0
    nonzero = 0
    for tensor in task_vector.values():
        flat = tensor.flatten()
        total += flat.numel()
        nonzero += (flat.abs() > 1e-10).sum().item()
    
    sparsity_pct = 100.0 * (total - nonzero) / total
    
    # Don't adapt for balanced sparsity
    if sparsity_threshold < sparsity_pct < (100 - sparsity_threshold):
        return base_range
    
    # For very sparse vectors, consider smaller coefficients
    if sparsity_pct > (100 - sparsity_threshold):
        return [c * 0.5 for c in base_range]
    
    # For dense vectors, consider larger coefficients
    return [c * 1.5 for c in base_range]


def estimate_memory_usage(model, task_vectors_count):
    """
    Estimate memory usage for task vectors and models.
    
    Args:
        model: PyTorch model.
        task_vectors_count: Number of task vectors to keep in memory.
        
    Returns:
        Estimated memory in MB.
    """
    model_params = sum(p.numel() for p in model.parameters())
    bytes_per_param = 4  # float32
    model_mb = model_params * bytes_per_param / (1024 * 1024)
    
    # Task vectors have same size as model
    tv_mb = model_mb * task_vectors_count
    
    # Evaluation overhead
    overhead_mb = 100  # General overhead
    
    total_mb = model_mb + tv_mb + overhead_mb
    return {
        'model_mb': model_mb,
        'task_vectors_mb': tv_mb,
        'total_mb': total_mb,
    }


def profile_coefficient_search(coef_range, num_coefs_to_profile=5):
    """
    Profile which coefficients are most likely to matter.
    
    Strategy: Test coefficients at different scales to find promising region.
    
    Args:
        coef_range: Full coefficient range.
        num_coefs_to_profile: How many coefficients to sample for profiling.
        
    Returns:
        List of profiling coefficients to evaluate.
    """
    if len(coef_range) <= num_coefs_to_profile:
        return coef_range
    
    # Sample from beginning, middle, and end
    step = len(coef_range) // (num_coefs_to_profile - 1)
    profile_indices = [i * step for i in range(num_coefs_to_profile)]
    profile_indices = [min(i, len(coef_range) - 1) for i in profile_indices]
    
    return [coef_range[i] for i in sorted(set(profile_indices))]
