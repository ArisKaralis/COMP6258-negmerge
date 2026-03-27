"""
Enhanced main script with checkpointing, focus mode, and performance improvements.

Usage examples:
    python main_enhanced.py                    # Run all trials
    python main_enhanced.py --trial 0          # Run only trial 0
    python main_enhanced.py --methods negmerge # Run only NegMerge
    python main_enhanced.py --skip-pretrain    # Use cached pre-trained model
    python main_enhanced.py --cache-tvs        # Cache task vectors for reuse
    python main_enhanced.py --focus-retrain    # Only evaluate retrain baseline
"""

import argparse
import copy
import json
import os
import random
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpointing import CheckpointManager
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


def run_trial(trial_idx, pretrained_model, checkpoint_mgr, args):
    """
    Run one independent trial with checkpointing and focus options.
    
    Args:
        trial_idx: Trial index (0-based).
        pretrained_model: Pretrained ResNet-18 model.
        checkpoint_mgr: CheckpointManager instance.
        args: Command-line arguments with focus settings.
        
    Returns:
        Tuple of (trial_results, retrain_ref).
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

    # Early exit for focus-retrain mode
    if args.focus_retrain:
        print("    [Focus mode: retrain only]")
        retrain_ref = dict(acc_dr=retrain_acc_dr, acc_df=retrain_acc_df,
                           acc_dtest=retrain_acc_dtest, mia=retrain_mia)
        return {}, retrain_ref

    # ── 2. Build 27-model pool (fine-tune on forget set) ──────────────────────
    print("  [2/4] Building 27-model pool (fine-tune on forget set) ...")
    
    # Try to load cached task vectors
    task_vectors = None
    if args.cache_tvs:
        task_vectors = checkpoint_mgr.load_task_vectors(trial_idx)
    
    if task_vectors is None:
        task_vectors = []
        configs = list(product(EPOCHS_LIST, WD_LIST, LS_LIST))  # 27 configs
        for (ep, wd, ls) in tqdm(configs, desc="  Fine-tuning pool", leave=False):
            m = clone_model(pretrained_model)
            train_model(m, forget_train_loader, epochs=ep, weight_decay=wd,
                        label_smoothing=ls, desc=f"  ft e={ep} wd={wd} ls={ls}")
            fsd = copy.deepcopy(m.state_dict())
            task_vectors.append(compute_task_vector(pretrained_sd, fsd))
        
        # Save task vectors if caching is enabled
        if args.cache_tvs:
            checkpoint_mgr.save_task_vectors(trial_idx, task_vectors)

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

    # Filter methods based on focus
    methods_to_run = {
        "NegMerge (ours)": negmerge,
        "Uniform Merge": uniform_merge,
        "TIES-Merging": ties_merging,
        "MagMax": magmax,
        "Task Arithmetic†": None,  # Special handling
    }
    
    if args.methods:
        methods_to_run = {k: v for k, v in methods_to_run.items() 
                         if any(m in k for m in args.methods)}

    # Compute sparsity for all merged task vectors
    if not args.skip_sparsity:
        print("\n  [Sparsity Analysis]")
        sparsity_results = {}
        
        for method_name, merge_fn in methods_to_run.items():
            if method_name == "Task Arithmetic†":
                continue
            tv = merge_fn(pretrained_sd, task_vectors)
            sp = compute_sparsity(tv)
            sparsity_results[method_name] = sp
            print(f"    {method_name:<26}: {sp['sparsity_pct']:.2f}% sparse "
                  f"({sp['nonzero_params']:,}/{sp['total_params']:,} params active)")
        print()

    # Setup coefficient search threshold
    pretrained_eval_m = load_model_with_sd(pretrained_model, pretrained_sd)
    retain_pretrained = evaluate(pretrained_eval_m, retain_loader)
    threshold = 0.95 * retain_pretrained

    # ── NegMerge ──────────────────────────────────────────────────────────────
    if "NegMerge (ours)" in methods_to_run:
        nm_tv = negmerge(pretrained_sd, task_vectors)
        best_coef_nm, _ = select_best_coef(
            pretrained_sd, nm_tv, coef_range, pretrained_model,
            retain_loader, forget_loader, threshold)
        nm_sd = apply_task_vector(pretrained_sd, nm_tv, coef=-best_coef_nm)
        nm_m = load_model_with_sd(pretrained_model, nm_sd)
        record("NegMerge (ours)", nm_m)
        checkpoint_mgr.save_best_coefs(trial_idx, "NegMerge", {"coef": best_coef_nm})

    # ── Task Arithmetic: Single Best Model ────────────────────────────────────
    if "Task Arithmetic†" in methods_to_run:
        best_tv, best_coef_ta = find_best_single_tv(
            pretrained_sd, task_vectors, pretrained_model,
            forget_loader, retain_loader, coef_range)
        ta_sd = apply_task_vector(pretrained_sd, best_tv, coef=-best_coef_ta)
        ta_m = load_model_with_sd(pretrained_model, ta_sd)
        record("Task Arithmetic†", ta_m)
        checkpoint_mgr.save_best_coefs(trial_idx, "TaskArithmetic", {"coef": best_coef_ta})

    # ── Uniform Merge ─────────────────────────────────────────────────────────
    if "Uniform Merge" in methods_to_run:
        um_tv = uniform_merge(task_vectors)
        best_coef_um, _ = select_best_coef(
            pretrained_sd, um_tv, coef_range, pretrained_model,
            retain_loader, forget_loader, threshold)
        um_sd = apply_task_vector(pretrained_sd, um_tv, coef=-best_coef_um)
        um_m = load_model_with_sd(pretrained_model, um_sd)
        record("Uniform Merge", um_m)
        checkpoint_mgr.save_best_coefs(trial_idx, "UniformMerge", {"coef": best_coef_um})

    # ── TIES-Merging ──────────────────────────────────────────────────────────
    if "TIES-Merging" in methods_to_run:
        ties_tv = ties_merging(task_vectors)
        best_coef_ties, _ = select_best_coef(
            pretrained_sd, ties_tv, coef_range, pretrained_model,
            retain_loader, forget_loader, threshold)
        ties_sd = apply_task_vector(pretrained_sd, ties_tv, coef=-best_coef_ties)
        ties_m = load_model_with_sd(pretrained_model, ties_sd)
        record("TIES-Merging", ties_m)
        checkpoint_mgr.save_best_coefs(trial_idx, "TIES", {"coef": best_coef_ties})

    # ── MagMax ────────────────────────────────────────────────────────────────
    if "MagMax" in methods_to_run:
        mm_tv = magmax(task_vectors)
        best_coef_mm, _ = select_best_coef(
            pretrained_sd, mm_tv, coef_range, pretrained_model,
            retain_loader, forget_loader, threshold)
        mm_sd = apply_task_vector(pretrained_sd, mm_tv, coef=-best_coef_mm)
        mm_m = load_model_with_sd(pretrained_model, mm_sd)
        record("MagMax", mm_m)
        checkpoint_mgr.save_best_coefs(trial_idx, "MagMax", {"coef": best_coef_mm})

    # ── 4. Package trial results ───────────────────────────────────────────────
    retrain_ref = dict(acc_dr=retrain_acc_dr, acc_df=retrain_acc_df,
                       acc_dtest=retrain_acc_dtest, mia=retrain_mia)
    return results, retrain_ref


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="NegMerge CIFAR-10 experiments with checkpointing and focus mode"
    )
    parser.add_argument("--trial", type=int, default=None,
                       help="Run specific trial (0-indexed). If None, run all trials.")
    parser.add_argument("--methods", nargs="+", default=None,
                       help="Run specific methods (e.g., negmerge ties magmax)")
    parser.add_argument("--skip-pretrain", action="store_true",
                       help="Skip pre-training, use cached model")
    parser.add_argument("--cache-tvs", action="store_true",
                       help="Cache task vectors for reuse across runs")
    parser.add_argument("--skip-sparsity", action="store_true",
                       help="Skip sparsity analysis")
    parser.add_argument("--focus-retrain", action="store_true",
                       help="Only evaluate retrain baseline (quick test)")
    args = parser.parse_args()

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

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(CKPT_DIR)

    # Determine trials to run
    if args.trial is not None:
        trial_indices = [args.trial]
        print(f"\nRunning TRIAL {args.trial} only (focus mode)")
    else:
        trial_indices = range(NUM_TRIALS)
        print(f"\nRunning all {NUM_TRIALS} trials")

    if args.methods:
        print(f"Focus methods: {', '.join(args.methods)}")
    if args.focus_retrain:
        print("Focus mode: retrain baseline only")
    if args.cache_tvs:
        print("Task vector caching enabled")

    pool_size = len(list(product(EPOCHS_LIST, WD_LIST, LS_LIST)))
    print(f"Model pool size: {pool_size} models per trial\n")

    # ── Pre-train once on full CIFAR-10 ──────────────────────────────────────
    if not args.skip_pretrain:
        ckpt_path = os.path.join(CKPT_DIR, "pretrained_resnet18.pt")
        if os.path.exists(ckpt_path):
            print(f"Loading pre-trained model from {ckpt_path}")
            pretrained_model = build_resnet18()
            state_dict = checkpoint_mgr.load_pretrained_model(ckpt_path)
            pretrained_model.load_state_dict(state_dict)
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
            checkpoint_mgr.save_pretrained_model(pretrained_model)
    else:
        print("Loading cached pre-trained model...")
        pretrained_model = build_resnet18()
        state_dict = checkpoint_mgr.load_pretrained_model()
        if state_dict is not None:
            pretrained_model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError("Pre-trained model not found and --skip-pretrain specified")

    # ── Run trials ────────────────────────────────────────────────────────────
    all_results = []
    all_retrains = []

    for trial in trial_indices:
        trial_results, retrain_ref = run_trial(trial, pretrained_model, checkpoint_mgr, args)
        all_results.append(trial_results)
        all_retrains.append(retrain_ref)

        # Save checkpoint
        if trial_results:  # Only save if methods were actually run
            checkpoint_mgr.save_trial_checkpoint(trial, trial_results, retrain_ref)

    # ── Print final table ─────────────────────────────────────────────────────
    if all_results and any(all_results):  # Only print if we have methods results
        print_table(all_results, all_retrains)
    else:
        print("\nNo method results to report (focus mode or retrain-only)")
        print(f"Retrain baseline accuracy: {all_retrains[0]['acc_dtest']:.2f}%")


if __name__ == "__main__":
    main()
