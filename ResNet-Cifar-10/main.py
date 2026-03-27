"""
Main orchestration for NegMerge CIFAR-10 experiments.

NegMerge - Table 2 Reproduction
================================
Unlearning Performance for 10% Random Data Forgetting on CIFAR-10 using ResNet-18.

Reproduces: NegMerge row + Task Arithmetic baselines (Single Best, Uniform Merge,
TIES-Merging, MagMax).  Starred (*) rows from Fan et al. 2024 are inserted as
constants so the final table matches the paper exactly.

Usage:
    python main.py

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
"""

import copy
import json
import os
import random
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, CKPT_DIR, DATA_DIR, DEVICE, EPOCHS_LIST,
                    FORGET_RATIO, LS_LIST, NUM_TRIALS, NUM_WORKERS, PREFETCH,
                    SEED, WD_LIST, print_device_diagnostics)
from data import get_datasets, get_transforms, make_loader
from evaluation import compute_sparsity, mia_efficacy
from merging import magmax, negmerge, ties_merging, uniform_merge
from models import build_resnet18, clone_model
from reporting import print_table
from training import evaluate, train_model
from utils import (apply_task_vector, compute_task_vector,
                   load_model_with_sd, select_best_coef)


def find_best_single_tv(pretrained_sd, task_vectors, base_model,
                        forget_loader, retain_loader, coef_range,
                        retain_threshold_pct=0.95):
    """Task Arithmetic single-best-model selection."""
    from utils import build_state_dict_from_task_vector
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
                best_tv = tv
                best_coef = coef

    return best_tv, best_coef


def run_trial(trial_idx, pretrained_model):
    """
    Run one independent trial with all unlearning methods.
    
    Args:
        trial_idx: Trial index (0-based).
        pretrained_model: Pretrained ResNet-18 model.
        
    Returns:
        Tuple of (trial_results, retrain_ref) where:
        - trial_results: Dict {method_name: {acc_dr, acc_df, acc_dtest, mia}}
        - retrain_ref: Dict {acc_dr, acc_df, acc_dtest, mia}
    """
    print(f"\n{'='*60}")
    print(f"  TRIAL {trial_idx + 1} / {NUM_TRIALS}")
    print(f"{'='*60}")

    seed = SEED + trial_idx * 100
    _, forget_ds, retain_ds, test_ds = get_datasets(seed, FORGET_RATIO)

    forget_loader = make_loader(forget_ds, shuffle=False)
    retain_loader = make_loader(retain_ds, shuffle=False)
    test_loader = make_loader(test_ds, shuffle=False)
    forget_train_loader = make_loader(forget_ds, shuffle=True)

    pretrained_sd = copy.deepcopy(pretrained_model.state_dict())

    # ── 1. Retrain baseline ────────────────────────────────────────────────────
    print("  [1/4] Training retrain baseline ...")
    retrain_m = build_resnet18()
    retrain_loader_train = make_loader(retain_ds, shuffle=True)
    train_model(retrain_m, retrain_loader_train, epochs=60, desc="  Retrain")

    retrain_acc_dr = evaluate(retrain_m, retain_loader)
    retrain_acc_df = evaluate(retrain_m, forget_loader)
    retrain_acc_dtest = evaluate(retrain_m, test_loader)
    retrain_mia = mia_efficacy(retrain_m, forget_loader, retain_loader)

    print(f"    Retrain → Dr={retrain_acc_dr:.2f}  Df={retrain_acc_df:.2f}  "
          f"Dtest={retrain_acc_dtest:.2f}  MIA={retrain_mia:.2f}")

    # ── 2. Build 27-model pool (fine-tune on forget set) ──────────────────────
    print("  [2/4] Building 27-model pool (fine-tune on forget set) ...")
    task_vectors = []
    configs = list(product(EPOCHS_LIST, WD_LIST, LS_LIST))  # 27 configs
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
        """Record evaluation results for a method."""
        acc_dr = evaluate(model, retain_loader)
        acc_df = evaluate(model, forget_loader)
        acc_dtest = evaluate(model, test_loader)
        mia = mia_efficacy(model, forget_loader, retain_loader)
        results[name] = dict(acc_dr=acc_dr, acc_df=acc_df,
                             acc_dtest=acc_dtest, mia=mia)
        print(f"    {name:<30} Dr={acc_dr:.2f}  Df={acc_df:.2f}  "
              f"Dtest={acc_dtest:.2f}  MIA={mia:.2f}")

    # Compute sparsity for all merged task vectors
    print("\n  [Sparsity Analysis]")
    sparsity_results = {}

    nm_tv = negmerge(pretrained_sd, task_vectors)
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
    pretrained_eval_m = load_model_with_sd(pretrained_model, pretrained_sd)
    retain_pretrained = evaluate(pretrained_eval_m, retain_loader)
    threshold = 0.95 * retain_pretrained

    best_coef_nm, _ = select_best_coef(
        pretrained_sd, nm_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)

    nm_sd = apply_task_vector(pretrained_sd, nm_tv, coef=-best_coef_nm)
    nm_m = load_model_with_sd(pretrained_model, nm_sd)
    record("NegMerge (ours)", nm_m)

    # ── Task Arithmetic: Single Best Model ────────────────────────────────────
    best_tv, best_coef_ta = find_best_single_tv(
        pretrained_sd, task_vectors, pretrained_model,
        forget_loader, retain_loader, coef_range)
    ta_sd = apply_task_vector(pretrained_sd, best_tv, coef=-best_coef_ta)
    ta_m = load_model_with_sd(pretrained_model, ta_sd)
    record("Task Arithmetic†", ta_m)

    # ── Uniform Merge ─────────────────────────────────────────────────────────
    best_coef_um, _ = select_best_coef(
        pretrained_sd, um_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    um_sd = apply_task_vector(pretrained_sd, um_tv, coef=-best_coef_um)
    um_m = load_model_with_sd(pretrained_model, um_sd)
    record("Uniform Merge", um_m)

    # ── TIES-Merging ──────────────────────────────────────────────────────────
    best_coef_ties, _ = select_best_coef(
        pretrained_sd, ties_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    ties_sd = apply_task_vector(pretrained_sd, ties_tv, coef=-best_coef_ties)
    ties_m = load_model_with_sd(pretrained_model, ties_sd)
    record("TIES-Merging", ties_m)

    # ── MagMax ────────────────────────────────────────────────────────────────
    best_coef_mm, _ = select_best_coef(
        pretrained_sd, mm_tv, coef_range, pretrained_model,
        retain_loader, forget_loader, threshold)
    mm_sd = apply_task_vector(pretrained_sd, mm_tv, coef=-best_coef_mm)
    mm_m = load_model_with_sd(pretrained_model, mm_sd)
    record("MagMax", mm_m)

    # ── 4. Package trial results ───────────────────────────────────────────────
    retrain_ref = dict(acc_dr=retrain_acc_dr, acc_df=retrain_acc_df,
                       acc_dtest=retrain_acc_dtest, mia=retrain_mia)
    return results, retrain_ref


def main():
    """Main entry point for experiments."""
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    print("Device:", DEVICE)
    print_device_diagnostics(DEVICE)
    print(f"DataLoader workers: {NUM_WORKERS}, prefetch_factor: {PREFETCH}, batch_size: {BATCH_SIZE}")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    pool_size = len(list(product(EPOCHS_LIST, WD_LIST, LS_LIST)))
    print(f"Running {NUM_TRIALS} trial(s) with {pool_size} models in pool each.\n")

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
        from torchvision import datasets
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
    all_results = []
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
