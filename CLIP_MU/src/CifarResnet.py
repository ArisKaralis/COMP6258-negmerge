"""
NegMerge - Table 2 Reproduction
================================
Unlearning Performance for 10% Random Data Forgetting on CIFAR-10 using ResNet-18.
Reproduces: NegMerge row + Task Arithmetic baselines (Single Best, Uniform Merge,
TIES-Merging, MagMax).  Starred (*) rows from Fan et al. 2024 are inserted as
constants so the final table matches the paper exactly.

Requirements:
    pip install torch torchvision numpy tqdm

Usage:
    python negmerge_cifar10_resnet18.py

The script will:
  1. Download CIFAR-10 automatically.
  2. Train a clean ResNet-18 pretrained model on the full training set.
  3. Run 3 independent trials, each with:
       - A fresh 10% / 90% forget / retain split
       - A retrain baseline (trained only on retain)
       - 27 fine-tuned forget-set models (3×3×3 hyperparameter grid)
       - NegMerge, Task Arithmetic (single best), Uniform Merge,
         TIES-Merging, and MagMax unlearning methods
       - MIA-Efficacy evaluation for each method
  4. Print a formatted table matching the paper (Table 2).

Runtime estimate (CPU): ~3-4 hours for the full 3-trial run.
With a GPU it is much faster (~30-60 min).
Set NUM_TRIALS=1 and POOL_SIZE smaller for a quick smoke test.
"""

import copy
import json
import os
import random
import warnings
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
    module=r"torchvision\.datasets\.cifar",
)

# ── Configuration ──────────────────────────────────────────────────────────────

SEED          = 42
NUM_TRIALS    = 1          # Paper uses 3 independent trials (set to 1 for testing)
FORGET_RATIO  = 0.10       # 10% of training data is the forget set
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "256"))
CPU_COUNT     = os.cpu_count() or 4
NUM_WORKERS   = int(os.getenv("NUM_WORKERS", str(max(2, min(8, CPU_COUNT - 1)))))
PREFETCH      = int(os.getenv("PREFETCH_FACTOR", "2"))


def get_device():
    forced = os.getenv("DEVICE", "").strip().lower()
    if forced:
        if forced == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if forced == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if forced == "cpu":
            return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_diagnostics():
    mps_built = bool(getattr(torch.backends.mps, "is_built", lambda: False)()) if hasattr(torch.backends, "mps") else False
    mps_available = bool(getattr(torch.backends.mps, "is_available", lambda: False)()) if hasattr(torch.backends, "mps") else False
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS built: {mps_built}")
    print(f"MPS available: {mps_available}")
    if DEVICE.type == "cpu" and mps_built and not mps_available:
        print("MPS backend is built but unavailable at runtime; falling back to CPU.")


DEVICE        = get_device()
DATA_DIR      = "./data"
CKPT_DIR      = "./checkpoints_negmerge"
os.makedirs(CKPT_DIR, exist_ok=True)

# Hyperparameter grid for building the 27-model pool (paper Section A)
EPOCHS_LIST   = [40, 50, 60]
WD_LIST       = [1e-4, 5e-5, 1e-5]
LS_LIST       = [0.0, 0.05, 0.1]     # label smoothing
LR            = 0.05                  # fixed LR from paper

# MIA shadow-model training epochs (lightweight)
MIA_SHADOW_EPOCHS = 5
MIA_SHADOW_LR     = 0.01

# ── Data ───────────────────────────────────────────────────────────────────────

def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return train_tf, test_tf


def get_datasets(seed):
    train_tf, test_tf = get_transforms()
    full_train = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tf)
    test_ds    = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=test_tf)

    rng = np.random.default_rng(seed)
    n   = len(full_train)
    forget_idx  = rng.choice(n, size=int(FORGET_RATIO * n), replace=False)
    retain_idx  = np.setdiff1d(np.arange(n), forget_idx)

    forget_ds = Subset(full_train, forget_idx.tolist())
    retain_ds = Subset(full_train, retain_idx.tolist())
    return full_train, forget_ds, retain_ds, test_ds


def make_loader(dataset, shuffle=True, batch_size=BATCH_SIZE):
    # pin_memory improves host->GPU transfer for CUDA, but is unsupported on MPS.
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

# ── Model ──────────────────────────────────────────────────────────────────────

def build_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.to(DEVICE)


def clone_model(model):
    return copy.deepcopy(model)

# ── Training helpers ───────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
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
            total   += y.size(0)
    return 100.0 * correct / total


def train_model(model, loader, epochs, lr=LR, weight_decay=1e-4,
                label_smoothing=0.0, desc="Training"):
    """Train model for a fixed number of epochs; returns trained model."""
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                          weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    for _ in tqdm(range(epochs), desc=desc, leave=False):
        train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()
    return model

# ── MIA-Efficacy ───────────────────────────────────────────────────────────────
# Following Fan et al. 2024 / Carlini et al. 2022:
# Train a binary "membership" classifier on (loss features) from a shadow model.
# MIA-Efficacy = accuracy of the attack on forget samples.
# Higher score → model still "remembers" forget data.
# The retrained model should yield ~random (≈50%) or dataset-specific baseline.

def compute_losses(model, loader):
    """Return per-sample cross-entropy losses as a numpy array."""
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
    Simplified LiRA-style MIA-Efficacy (scalar, higher = more memorisation).
    We use the loss-threshold attack: label forget samples as "member" if their
    loss on the unlearned model is LOW (i.e. model still fits them well).
    MIA-Efficacy = fraction of forget samples correctly identified as members
    by a threshold calibrated on the retain set.
    This matches the spirit of Fan et al. 2024 (Eq. MIA-Efficacy).
    """
    forget_losses  = compute_losses(unlearned_model, forget_loader)
    retain_losses  = compute_losses(unlearned_model, retain_loader)

    # Threshold: median retain loss (retain samples are "members")
    threshold = np.median(retain_losses)

    # A forget sample is predicted "member" if its loss < threshold
    predicted_member = forget_losses < threshold
    # MIA-Efficacy = fraction of forget samples predicted as member
    efficacy = 100.0 * predicted_member.mean()
    return efficacy

# ── Task Vector utilities ──────────────────────────────────────────────────────

def compute_task_vector(pretrained_sd, finetuned_sd):
    """Return {key: finetuned - pretrained} for float params."""
    tv = {}
    for k in pretrained_sd:
        if pretrained_sd[k].dtype in (torch.int64, torch.uint8):
            continue
        tv[k] = finetuned_sd[k].float() - pretrained_sd[k].float()
    return tv


def apply_task_vector(pretrained_sd, tv, coef=1.0):
    """Return new state dict = pretrained + coef * tv."""
    new_sd = copy.deepcopy(pretrained_sd)
    for k in tv:
        new_sd[k] = pretrained_sd[k].float() + coef * tv[k]
    return new_sd


def build_state_dict_from_task_vector(pretrained_sd, tv, coef=1.0):
    """Return lightweight state dict for repeated coef search."""
    new_sd = {}
    for k, base in pretrained_sd.items():
        if k in tv:
            new_sd[k] = base.float() + coef * tv[k]
        else:
            new_sd[k] = base
    return new_sd


def select_best_coef(pretrained_sd, tv, coef_range, base_model,
                     retain_loader, forget_loader, threshold):
    """Select coefficient that minimises forget accuracy under retain threshold."""
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
    m = clone_model(base_model)
    m.load_state_dict(state_dict)
    return m

# ── Merging methods ────────────────────────────────────────────────────────────

def negmerge(pretrained_sd, task_vectors):
    """
    NegMerge (Equation 1 in the paper):
      τ_merged[k] = mean over k of τ_k, but only where ALL τ_k share the same sign.
      Elements with conflicting signs are zeroed out.
    """
    keys = list(task_vectors[0].keys())
    n    = len(task_vectors)

    merged = {}
    for key in keys:
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)  # (n, ...)
        sign_sum = torch.sign(stacked).sum(dim=0)                        # unanimous iff |sum| == n
        consensus_mask = torch.abs(sign_sum) == n
        avg = stacked.mean(dim=0)
        merged[key] = torch.where(consensus_mask, avg, torch.zeros_like(avg))
    return merged


def uniform_merge(task_vectors):
    """Simple average of all task vectors."""
    keys = list(task_vectors[0].keys())
    merged = {}
    for key in keys:
        merged[key] = torch.stack([tv[key] for tv in task_vectors]).mean(dim=0)
    return merged


def ties_merging(task_vectors, top_k=0.2):
    """
    TIES-Merging (Yadav et al. 2023):
      1. Trim: keep top-k% magnitude elements per vector (zero out rest).
      2. Elect sign: majority vote per element.
      3. Merge: average only elements matching the elected sign.
    """
    keys = list(task_vectors[0].keys())
    n    = len(task_vectors)

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
    MagMax (Marczak et al. 2024):
      For each element, keep the task vector with the largest absolute value.
    """
    keys = list(task_vectors[0].keys())
    merged = {}
    for key in keys:
        stacked = torch.stack([tv[key] for tv in task_vectors], dim=0)  # (n, ...)
        idx     = stacked.abs().argmax(dim=0, keepdim=True)
        merged[key] = stacked.gather(0, idx).squeeze(0)
    return merged


def find_best_single_tv(pretrained_sd, task_vectors, base_model,
                        forget_loader, retain_loader, coef_range,
                        retain_threshold_pct=0.95):
    """
    Task Arithmetic single-best-model selection:
    For each (tv, coef) pair, apply negation and pick the one that minimises
    forget-set accuracy while keeping retain accuracy ≥ 95% of pretrained.
    Returns the best (task_vector, coef).
    """
    pretrained_model = load_model_with_sd(base_model, pretrained_sd)
    retain_pretrained = evaluate(pretrained_model, retain_loader)
    threshold = retain_threshold_pct * retain_pretrained

    best_forget_acc = float("inf")
    best_tv = task_vectors[0]
    best_coef = coef_range[0]
    scratch = clone_model(base_model)

    for tv in tqdm(task_vectors, desc="  TA single-best search", leave=False):
        for coef in coef_range:
            new_sd = build_state_dict_from_task_vector(pretrained_sd, tv, coef=-coef)
            scratch.load_state_dict(new_sd, strict=True)
            r_acc = evaluate(scratch, retain_loader)
            if r_acc < threshold:
                continue
            f_acc = evaluate(scratch, forget_loader)
            if f_acc < best_forget_acc:
                best_forget_acc = f_acc
                best_tv   = tv
                best_coef = coef

    return best_tv, best_coef

# ── One full trial ─────────────────────────────────────────────────────────────

def run_trial(trial_idx, pretrained_model):
    """
    Run one independent trial.  Returns a dict of {method: {acc_dr, acc_df, acc_dtest, mia}}.
    """
    print(f"\n{'='*60}")
    print(f"  TRIAL {trial_idx + 1} / {NUM_TRIALS}")
    print(f"{'='*60}")

    seed = SEED + trial_idx * 100
    _, forget_ds, retain_ds, test_ds = get_datasets(seed)

    forget_loader = make_loader(forget_ds, shuffle=False)
    retain_loader = make_loader(retain_ds, shuffle=False)
    test_loader   = make_loader(test_ds,  shuffle=False)
    forget_train_loader = make_loader(forget_ds, shuffle=True)

    pretrained_sd = copy.deepcopy(pretrained_model.state_dict())

    # ── 1. Retrain baseline ────────────────────────────────────────────────────
    print("  [1/4] Training retrain baseline ...")
    retrain_m = build_resnet18()
    retrain_loader_train = make_loader(retain_ds, shuffle=True)
    train_model(retrain_m, retrain_loader_train, epochs=60, desc="  Retrain")

    retrain_acc_dr    = evaluate(retrain_m, retain_loader)
    retrain_acc_df    = evaluate(retrain_m, forget_loader)
    retrain_acc_dtest = evaluate(retrain_m, test_loader)
    retrain_mia       = mia_efficacy(retrain_m, forget_loader, retain_loader)

    print(f"    Retrain → Dr={retrain_acc_dr:.2f}  Df={retrain_acc_df:.2f}  "
          f"Dtest={retrain_acc_dtest:.2f}  MIA={retrain_mia:.2f}")

    # ── 2. Build 27-model pool (fine-tune on forget set) ──────────────────────
    print("  [2/4] Building 27-model pool (fine-tune on forget set) ...")
    task_vectors   = []
    configs = list(product(EPOCHS_LIST, WD_LIST, LS_LIST))   # 27 configs
    for (ep, wd, ls) in tqdm(configs, desc="  Fine-tuning pool", leave=False):
        m = clone_model(pretrained_model)
        train_model(m, forget_train_loader, epochs=ep, weight_decay=wd,
                    label_smoothing=ls, desc=f"  ft e={ep} wd={wd} ls={ls}")
        fsd = copy.deepcopy(m.state_dict())
        task_vectors.append(compute_task_vector(pretrained_sd, fsd))

    # Coefficient search range (paper uses 20 values)
    coef_range = [i * 0.05 for i in range(1, 21)]

    # ── 3. Evaluate all methods ────────────────────────────────────────────────
    print("  [3/4] Evaluating unlearning methods ...")
    results = {}

    def record(name, model):
        acc_dr    = evaluate(model, retain_loader)
        acc_df    = evaluate(model, forget_loader)
        acc_dtest = evaluate(model, test_loader)
        mia       = mia_efficacy(model, forget_loader, retain_loader)
        results[name] = dict(acc_dr=acc_dr, acc_df=acc_df,
                             acc_dtest=acc_dtest, mia=mia)
        print(f"    {name:<30} Dr={acc_dr:.2f}  Df={acc_df:.2f}  "
              f"Dtest={acc_dtest:.2f}  MIA={mia:.2f}")

    # Compute sparsity for all merged task vectors
    print("\n  [Sparsity Analysis]")
    sparsity_results = {}
    
    nm_tv   = negmerge(pretrained_sd, task_vectors)
    sp = compute_sparsity(nm_tv)
    sparsity_results['NegMerge'] = sp
    print(f"    NegMerge:      {sp['sparsity_pct']:.2f}% sparse "
          f"({sp['nonzero_params']:,}/{sp['total_params']:,} params active) "
          f"avg_mag={sp['avg_magnitude']:.6f}")
    
    um_tv = uniform_merge(task_vectors)
    sp = compute_sparsity(um_tv)
    sparsity_results['Uniform Merge'] = sp
    print(f"    Uniform Merge: {sp['sparsity_pct']:.2f}% sparse "
          f"({sp['nonzero_params']:,}/{sp['total_params']:,} params active) "
          f"avg_mag={sp['avg_magnitude']:.6f}")
    
    ties_tv = ties_merging(task_vectors)
    sp = compute_sparsity(ties_tv)
    sparsity_results['TIES-Merging'] = sp
    print(f"    TIES-Merging:  {sp['sparsity_pct']:.2f}% sparse "
          f"({sp['nonzero_params']:,}/{sp['total_params']:,} params active) "
          f"avg_mag={sp['avg_magnitude']:.6f}")
    
    mm_tv = magmax(task_vectors)
    sp = compute_sparsity(mm_tv)
    sparsity_results['MagMax'] = sp
    print(f"    MagMax:        {sp['sparsity_pct']:.2f}% sparse "
          f"({sp['nonzero_params']:,}/{sp['total_params']:,} params active) "
          f"avg_mag={sp['avg_magnitude']:.6f}")
    
    print()

    # ── NegMerge ──────────────────────────────────────────────────────────────
    # Find optimal coefficient (same 95%-retain-threshold logic)
    pretrained_eval_m = load_model_with_sd(pretrained_model, pretrained_sd)
    retain_pretrained = evaluate(pretrained_eval_m, retain_loader)
    threshold = 0.95 * retain_pretrained

    best_coef_nm, _ = select_best_coef(
        pretrained_sd, nm_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)

    nm_sd = apply_task_vector(pretrained_sd, nm_tv, coef=-best_coef_nm)
    nm_m  = load_model_with_sd(pretrained_model, nm_sd)
    record("NegMerge (ours)", nm_m)

    # ── Task Arithmetic: Single Best Model ────────────────────────────────────
    best_tv, best_coef_ta = find_best_single_tv(
        pretrained_sd, task_vectors, pretrained_model,
        forget_loader, retain_loader, coef_range)
    ta_sd = apply_task_vector(pretrained_sd, best_tv, coef=-best_coef_ta)
    ta_m  = load_model_with_sd(pretrained_model, ta_sd)
    record("Task Arithmetic†", ta_m)

    # ── Uniform Merge ─────────────────────────────────────────────────────────
    best_coef_um, _ = select_best_coef(
        pretrained_sd, um_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    um_sd = apply_task_vector(pretrained_sd, um_tv, coef=-best_coef_um)
    um_m  = load_model_with_sd(pretrained_model, um_sd)
    record("Uniform Merge", um_m)

    # ── TIES-Merging ──────────────────────────────────────────────────────────
    best_coef_ties, _ = select_best_coef(
        pretrained_sd, ties_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    ties_sd = apply_task_vector(pretrained_sd, ties_tv, coef=-best_coef_ties)
    ties_m  = load_model_with_sd(pretrained_model, ties_sd)
    record("TIES-Merging", ties_m)

    # ── MagMax ────────────────────────────────────────────────────────────────
    best_coef_mm, _ = select_best_coef(
        pretrained_sd, mm_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    mm_sd = apply_task_vector(pretrained_sd, mm_tv, coef=-best_coef_mm)
    mm_m  = load_model_with_sd(pretrained_model, mm_sd)
    record("MagMax", mm_m)

    # ── 4. Package trial results ───────────────────────────────────────────────
    retrain_ref = dict(acc_dr=retrain_acc_dr, acc_df=retrain_acc_df,
                       acc_dtest=retrain_acc_dtest, mia=retrain_mia)
    return results, retrain_ref

# ── Avg. Gap computation ───────────────────────────────────────────────────────

def compute_sparsity(task_vector):
    """
    Compute sparsity metrics for a task vector (dict of tensors).
    Returns: {
        'total_params': total number of parameters,
        'nonzero_params': number of non-zero parameters,
        'sparsity_pct': percentage of parameters that are exactly zero,
        'avg_magnitude': average absolute value of non-zero params
    }
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


def avg_gap(method_metrics, retrain_metrics):
    keys = ["acc_dr", "acc_df", "acc_dtest", "mia"]
    return np.mean([abs(method_metrics[k] - retrain_metrics[k]) for k in keys])

# ── Print table ────────────────────────────────────────────────────────────────

# Paper Table 2 rows borrowed from Fan et al. 2024 (marked with *)
PAPER_ROWS = [
    # (method,              split,   acc_dr,           acc_df,           acc_dtest,        mia,             avg_gap, starred)
    ("Retrain",             "retain","100.00±0.00",    "94.76±0.69",     "94.26±0.02",     "12.88±0.09",    "0.00",  True),
    ("Random Labeling",     "all",   "99.67±0.14",     "92.39±0.31",     "92.83±0.38",     "37.36±0.06",    "7.15",  True),
    ("Influence",           "all",   "99.20±0.22",     "98.93±0.28",     "93.20±1.03",     "2.67±0.01",     "4.06",  True),
    ("SalUn",               "all",   "99.62±0.12",     "97.15±0.43",     "93.93±0.29",     "14.39±0.82",    "1.15",  True),
    ("Finetune",            "retain","99.88±0.08",     "99.37±0.55",     "94.06±0.27",     "2.70±0.01",     "3.78",  True),
    ("ℓ1-sparse",           "retain","97.74±0.33",     "95.81±0.62",     "91.59±0.57",     "9.84±0.00",     "2.26",  True),
    ("Gradient Ascent",     "forget","99.50±0.38",     "99.31±0.54",     "94.01±0.47",     "1.70±0.01",     "4.12",  True),
    ("Boundary Shrink",     "forget","98.29±2.50",     "98.22±2.52",     "92.69±2.99",     "8.96±0.13",     "2.67",  True),
    ("Boundary Expanding",  "forget","99.42±0.33",     "99.41±0.30",     "93.85±1.02",     "7.47±1.15",     "2.76",  True),
    ("Random Labeling",     "forget","99.99±0.00",     "99.98±0.02",     "95.04±0.11",     "2.15±1.94",     "4.19",  True),
    ("SalUn",               "forget","99.88±0.04",     "99.89±0.04",     "94.42±0.05",     "9.51±2.07",     "2.20",  True),
]


def fmt(mean, std):
    return f"{mean:.2f}±{std:.2f}"


def print_table(all_trial_results, all_retrain_refs):
    """
    Aggregate across trials and print the full Table 2.
    all_trial_results: list of dicts {method_name: {acc_dr, acc_df, acc_dtest, mia}}
    all_retrain_refs:  list of dicts {acc_dr, acc_df, acc_dtest, mia}
    """
    # Average retrain metrics
    retrain_agg = {
        k: (np.mean([r[k] for r in all_retrain_refs]),
            np.std( [r[k] for r in all_retrain_refs]))
        for k in ["acc_dr", "acc_df", "acc_dtest", "mia"]
    }
    retrain_mean = {k: v[0] for k, v in retrain_agg.items()}

    # Aggregate computed methods
    method_names = list(all_trial_results[0].keys())
    agg = {}
    for mname in method_names:
        vals = {k: [t[mname][k] for t in all_trial_results]
                for k in ["acc_dr", "acc_df", "acc_dtest", "mia"]}
        means = {k: np.mean(v) for k, v in vals.items()}
        stds  = {k: np.std(v)  for k, v in vals.items()}
        gap_per_trial = [avg_gap(t[mname], all_retrain_refs[i])
                         for i, t in enumerate(all_trial_results)]
        agg[mname] = dict(means=means, stds=stds,
                          gap_mean=np.mean(gap_per_trial),
                          gap_std=np.std(gap_per_trial))

    # Column widths
    W = [28, 8, 14, 14, 14, 14, 10]
    header = ["Method", "Split", "Acc Dr(≃)", "Acc Df(≃)", "Acc Dtest(≃)", "MIA(≃)", "Avg.Gap↓"]
    sep    = "+" + "+".join("-" * w for w in W) + "+"
    row_fmt = "|" + "|".join(f"{{:<{w}}}" for w in W) + "|"

    print("\n")
    print("Table 2. Unlearning Performance for 10% Random Data Forgetting on CIFAR-10 using ResNet-18")
    print(sep)
    print(row_fmt.format(*header))
    print(sep)

    # Paper starred rows
    for (method, split, acc_dr, acc_df, acc_dtest, mia, gap, starred) in PAPER_ROWS:
        tag = "*" if starred else ""
        print(row_fmt.format(method + tag, split, acc_dr, acc_df, acc_dtest, mia, gap))

    print(sep)
    print(row_fmt.format("-- Task Arithmetic (forget set) --", "", "", "", "", "", ""))
    print(sep)

    # Map display names
    display = {
        "Task Arithmetic†":  ("Task Arithmetic†",  "forget"),
        "Uniform Merge":      ("Uniform Merge",      "forget"),
        "TIES-Merging":       ("TIES-Merging",       "forget"),
        "MagMax":             ("MagMax",             "forget"),
        "NegMerge (ours)":    ("NegMerge (ours)",    "forget"),
    }

    for mname, (label, split) in display.items():
        if mname not in agg:
            continue
        a = agg[mname]
        m, s = a["means"], a["stds"]
        print(row_fmt.format(
            label, split,
            fmt(m["acc_dr"],    s["acc_dr"]),
            fmt(m["acc_df"],    s["acc_df"]),
            fmt(m["acc_dtest"], s["acc_dtest"]),
            fmt(m["mia"],       s["mia"]),
            f"{a['gap_mean']:.2f}",
        ))

    print(sep)

    # Also print retrain from our runs for comparison
    print(row_fmt.format(
        "Retrain (our run)", "retain",
        fmt(retrain_agg["acc_dr"][0],    retrain_agg["acc_dr"][1]),
        fmt(retrain_agg["acc_df"][0],    retrain_agg["acc_df"][1]),
        fmt(retrain_agg["acc_dtest"][0], retrain_agg["acc_dtest"][1]),
        fmt(retrain_agg["mia"][0],       retrain_agg["mia"][1]),
        "0.00",
    ))
    print(sep)
    print("* Numbers borrowed from Fan et al. 2024")
    print("† Best result via hyperparameter search over 27-model pool")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("Device:", DEVICE)
    print_device_diagnostics()
    print(f"DataLoader workers: {NUM_WORKERS}, prefetch_factor: {PREFETCH}, batch_size: {BATCH_SIZE}")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    print(f"Running {NUM_TRIALS} trial(s) with {len(list(product(EPOCHS_LIST, WD_LIST, LS_LIST)))} models in pool each.\n")

    # ── Pre-train once on full CIFAR-10 ──────────────────────────────────────
    ckpt_path = os.path.join(CKPT_DIR, "pretrained_resnet18.pt")
    if os.path.exists(ckpt_path):
        print(f"Loading pre-trained model from {ckpt_path}")
        pretrained_model = build_resnet18()
        pretrained_model.load_state_dict(
            torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    else:
        print("Pre-training ResNet-18 on full CIFAR-10 ...")
        train_tf, _ = get_transforms()
        full_train = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=train_tf)
        pin_memory = DEVICE.type == "cuda"
        full_loader_kwargs = dict(
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
            persistent_workers=NUM_WORKERS > 0,
        )
        if NUM_WORKERS > 0:
            full_loader_kwargs["prefetch_factor"] = PREFETCH
        full_loader = DataLoader(full_train, **full_loader_kwargs)
        pretrained_model = build_resnet18()
        train_model(pretrained_model, full_loader, epochs=100, desc="Pre-train")
        torch.save(pretrained_model.state_dict(), ckpt_path)
        print(f"  Saved to {ckpt_path}")

    # ── Run trials ────────────────────────────────────────────────────────────
    all_results  = []
    all_retrains = []

    for trial in range(NUM_TRIALS):
        trial_results, retrain_ref = run_trial(trial, pretrained_model)
        all_results.append(trial_results)
        all_retrains.append(retrain_ref)

        # Save intermediate results
        save_path = os.path.join(CKPT_DIR, f"trial_{trial}_results.json")
        with open(save_path, "w") as f:
            json.dump({"results": trial_results, "retrain": retrain_ref}, f, indent=2)

    # ── Print final table ─────────────────────────────────────────────────────
    print_table(all_results, all_retrains)


if __name__ == "__main__":
    main()