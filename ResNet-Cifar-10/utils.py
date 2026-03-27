"""
Task vector utilities and helper functions.
"""

import copy

import numpy as np
import torch

from models import clone_model
from training import evaluate


def compute_task_vector(pretrained_sd, finetuned_sd):
    """
    Compute task vector as difference between finetuned and pretrained.
    
    Args:
        pretrained_sd: Pretrained model state dict.
        finetuned_sd: Finetuned model state dict.
        
    Returns:
        Task vector dictionary {key: finetuned - pretrained}.
    """
    tv = {}
    for k in pretrained_sd:
        if pretrained_sd[k].dtype in (torch.int64, torch.uint8):
            continue
        tv[k] = finetuned_sd[k].float() - pretrained_sd[k].float()
    return tv


def apply_task_vector(pretrained_sd, tv, coef=1.0):
    """
    Apply task vector to pretrained state dict.
    
    Args:
        pretrained_sd: Pretrained model state dict.
        tv: Task vector dictionary.
        coef: Coefficient to scale the task vector.
        
    Returns:
        New state dict = pretrained + coef * tv.
    """
    new_sd = copy.deepcopy(pretrained_sd)
    for k in tv:
        new_sd[k] = pretrained_sd[k].float() + coef * tv[k]
    return new_sd


def build_state_dict_from_task_vector(pretrained_sd, tv, coef=1.0):
    """
    Build state dict from task vector without deep copy of entire dict.
    
    Lightweight alternative to apply_task_vector for repeated coef search.
    
    Args:
        pretrained_sd: Pretrained model state dict.
        tv: Task vector dictionary.
        coef: Coefficient to scale the task vector.
        
    Returns:
        New state dict = pretrained + coef * tv.
    """
    new_sd = {}
    for k, base in pretrained_sd.items():
        if k in tv:
            new_sd[k] = base.float() + coef * tv[k]
        else:
            new_sd[k] = base
    return new_sd


def select_best_coef(pretrained_sd, tv, coef_range, base_model,
                     retain_loader, forget_loader, threshold):
    """
    Select coefficient that minimizes forget accuracy under retain threshold.
    
    Args:
        pretrained_sd: Pretrained model state dict.
        tv: Task vector to search over.
        coef_range: List of coefficients to try.
        base_model: Base model for loading state dicts.
        retain_loader: DataLoader for retain set.
        forget_loader: DataLoader for forget set.
        threshold: Minimum retain accuracy threshold.
        
    Returns:
        Tuple of (best_coef, best_forget_acc).
    """
    best_coef = coef_range[0]
    best_forget = float("inf")
    scratch = clone_model(base_model)

    for coef in coef_range:
        new_sd = build_state_dict_from_task_vector(pretrained_sd, tv, coef=-coef)
        scratch.load_state_dict(new_sd, strict=True)
        r_acc = evaluate(scratch, retain_loader)
        if r_acc < threshold:
            continue
        f_acc = evaluate(scratch, forget_loader)
        if f_acc < best_forget:
            best_forget = f_acc
            best_coef = coef

    return best_coef, best_forget


def load_model_with_sd(base_model, state_dict):
    """
    Load state dict into a cloned model.
    
    Args:
        base_model: Model to clone.
        state_dict: State dict to load.
        
    Returns:
        Model with loaded state dict.
    """
    m = clone_model(base_model)
    m.load_state_dict(state_dict)
    return m


def avg_gap(method_metrics, retrain_metrics):
    """
    Compute average gap between method and retrain baseline.
    
    Args:
        method_metrics: Dict with keys {acc_dr, acc_df, acc_dtest, mia}.
        retrain_metrics: Dict with keys {acc_dr, acc_df, acc_dtest, mia}.
        
    Returns:
        Average absolute difference across all metrics.
    """
    keys = ["acc_dr", "acc_df", "acc_dtest", "mia"]
    return np.mean([abs(method_metrics[k] - retrain_metrics[k]) for k in keys])
