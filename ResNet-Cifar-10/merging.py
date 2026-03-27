"""
Task vector merging methods: NegMerge, TIES-Merging, MagMax, Uniform Merge.
"""

import torch


def negmerge(pretrained_sd, task_vectors):
    """
    NegMerge (Equation 1 in the paper).
    
    Merges task vectors by averaging only elements where ALL vectors
    share the same sign. Elements with conflicting signs are zeroed out.
    
    Args:
        pretrained_sd: Ignored (kept for compatibility).
        task_vectors: List of task vector dictionaries.
        
    Returns:
        Merged task vector dictionary.
    """
    keys = list(task_vectors[0].keys())
    n = len(task_vectors)

    merged = {}
    for key in keys:
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)  # (n, ...)
        sign_sum = torch.sign(stacked).sum(dim=0)  # unanimous iff |sum| == n
        consensus_mask = torch.abs(sign_sum) == n
        avg = stacked.mean(dim=0)
        merged[key] = torch.where(consensus_mask, avg, torch.zeros_like(avg))
    return merged


def uniform_merge(task_vectors):
    """
    Simple average of all task vectors.
    
    Args:
        task_vectors: List of task vector dictionaries.
        
    Returns:
        Merged task vector dictionary (simple mean).
    """
    keys = list(task_vectors[0].keys())
    merged = {}
    for key in keys:
        merged[key] = torch.stack([tv[key] for tv in task_vectors]).mean(dim=0)
    return merged


def ties_merging(task_vectors, top_k=0.2):
    """
    TIES-Merging (Yadav et al. 2023).
    
    Three-step procedure:
    1. Trim: keep top-k% magnitude elements per vector (zero out rest).
    2. Elect sign: majority vote per element.
    3. Merge: average only elements matching the elected sign.
    
    Args:
        task_vectors: List of task vector dictionaries.
        top_k: Fraction of top-magnitude elements to keep (default: 0.2).
        
    Returns:
        Merged task vector dictionary.
    """
    keys = list(task_vectors[0].keys())
    n = len(task_vectors)

    # Step 1: Trim
    trimmed = []
    for tv in task_vectors:
        trimmed_tv = {}
        for key in keys:
            flat = tv[key].abs().flatten()
            threshold = torch.quantile(flat.float(), 1.0 - top_k)
            mask = tv[key].abs() >= threshold
            trimmed_tv[key] = tv[key] * mask
        trimmed.append(trimmed_tv)

    merged = {}
    for key in keys:
        stacked = torch.stack([t[key] for t in trimmed], dim=0)  # (n, ...)
        # Step 2: majority sign vote
        sign_vote = torch.sign(stacked.sum(dim=0))
        # Step 3: merge elements matching majority sign
        matching = torch.stack([
            torch.where(torch.sign(t[key]) == sign_vote, t[key],
                        torch.zeros_like(t[key]))
            for t in trimmed
        ], dim=0)
        merged[key] = matching.sum(dim=0) / (matching != 0).float().sum(dim=0).clamp(min=1)
    return merged


def magmax(task_vectors):
    """
    MagMax (Marczak et al. 2024).
    
    For each parameter element, selects the task vector with the largest
    absolute value magnitude.
    
    Args:
        task_vectors: List of task vector dictionaries.
        
    Returns:
        Merged task vector dictionary.
    """
    keys = list(task_vectors[0].keys())
    merged = {}
    for key in keys:
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)  # (n, ...)
        idx = stacked.abs().argmax(dim=0, keepdim=True)
        merged[key] = stacked.gather(0, idx).squeeze(0)
    return merged
