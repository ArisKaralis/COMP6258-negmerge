"""
Apply NegMerge to classifier models for CIFAR-10 machine unlearning.

Loads the pretrained checkpoint and all finetuned checkpoints from
finetune_forget.py, builds the sign-consensus task vector, and evaluates
unlearning quality (forget/retain/test accuracy and MIA AUC).

Also performs a coefficient search: sweeps alpha in [0, 1] and picks the
value that minimises forget accuracy while keeping retain accuracy above
95% of the pretrained baseline.

Usage (from repo root, with negmerge-env activated):
  python experiments/classifier_unlearning/run_negmerge.py \\
      --model resnet18 \\
      --pretrained-path checkpoints/classifier/resnet18/pretrained_best.pt \\
      --finetuned-dir checkpoints/classifier/resnet18/finetuned \\
      --split-dir checkpoints/classifier/resnet18 \\
      --results-dir results/classifier

Comparison note: run run_baseline.py with the same arguments to get the
Task Arithmetic baseline for direct comparison.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import gc
import glob
import json
import numpy as np
import torch

from experiments.classifier_unlearning.utils import (
    get_cifar10_loaders, get_model,
    ClassifierTaskVector, negmerge,
    accuracy, evaluate_unlearning,
)


def _clear_cache(device):
    """Release unused memory from the active device."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser('NegMerge Classifier Unlearning')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--pretrained-path', type=str,
                        default='checkpoints/classifier/resnet18/pretrained_best.pt')
    parser.add_argument('--finetuned-dir', type=str,
                        default='checkpoints/classifier/resnet18/finetuned',
                        help='Directory with per-HP finetuned .pt files from finetune_forget.py')
    parser.add_argument('--split-dir', type=str,
                        default='checkpoints/classifier/resnet18',
                        help='Directory with forget_indices.npy and retain_indices.npy')
    parser.add_argument('--results-dir', type=str, default='results/classifier')
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--forget-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-eval-points', type=int, default=21,
                        help='Number of alpha values to sweep in [0, 1]')
    parser.add_argument('--retain-threshold', type=float, default=0.95,
                        help='Retain accuracy must stay above this fraction of pretrained')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_model_with_sd(model_arch, state_dict, device):
    model = get_model(model_arch, num_classes=10)
    model.load_state_dict(state_dict)
    return model.to(device)


def sweep_coefficients(negmerge_tv, pretrained_sd, model_arch, loaders,
                       device, n_eval_points, retain_threshold):
    """
    Sweep alpha in [0,1]. For each alpha, apply the (negated) task vector
    and compute forget and retain accuracy.

    Memory strategy:
      - State dict is computed on CPU, then freed as soon as the model loads it.
      - Model is deleted and cache is flushed after every evaluation step.
      - At peak: one state dict (CPU) + one model (device) — never two models.

    Returns a list of dicts: [{alpha, forget_acc, retain_acc}, ...]
    """
    alphas = np.linspace(0.0, 1.0, n_eval_points)
    sweep = []
    for alpha in alphas:
        # Step 1: compute new state dict on CPU
        sd = negmerge_tv.apply_to(pretrained_sd, scaling_coef=float(alpha))

        # Step 2: build model architecture and load state dict — then immediately
        # free the dict so only the model exists on device
        model = get_model(model_arch, num_classes=10)
        model.load_state_dict(sd)
        del sd           # ← free CPU state dict before moving model to device
        model = model.to(device)

        # Step 3: evaluate
        forget_acc = accuracy(model, loaders['forget_eval'], device)
        retain_acc = accuracy(model, loaders['retain_eval'], device)

        # Step 4: free device memory before next iteration
        del model
        _clear_cache(device)

        sweep.append({'alpha': float(alpha), 'forget_acc': forget_acc, 'retain_acc': retain_acc})
        print(f"  alpha={alpha:.2f}  forget={forget_acc:.4f}  retain={retain_acc:.4f}")
    return sweep


def find_best_alpha(sweep, pretrained_retain_acc, retain_threshold):
    """
    Pick the alpha that minimises forget accuracy subject to:
      retain_accuracy >= retain_threshold * pretrained_retain_accuracy
    """
    threshold = retain_threshold * pretrained_retain_acc
    valid = [s for s in sweep if s['retain_acc'] >= threshold]
    if not valid:
        print(f"WARNING: No alpha met retain threshold {threshold:.4f}. "
              f"Using alpha with highest retain accuracy.")
        return max(sweep, key=lambda s: s['retain_acc'])['alpha']
    return min(valid, key=lambda s: s['forget_acc'])['alpha']


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}\nModel:  {args.model}")

    # Load pretrained checkpoint
    pretrained_sd = torch.load(
        args.pretrained_path, map_location='cpu', weights_only=True
    )
    print(f"Loaded pretrained from {args.pretrained_path}")

    # Load forget/retain split
    forget_idx = np.load(os.path.join(args.split_dir, 'forget_indices.npy'))
    retain_idx  = np.load(os.path.join(args.split_dir, 'retain_indices.npy'))

    data_dir = os.path.expanduser(args.data_dir)
    loaders = get_cifar10_loaders(
        data_dir=data_dir, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
        forget_indices=forget_idx, retain_indices=retain_idx,
    )

    # Pretrained model baseline
    print("\n=== Pretrained model (before unlearning) ===")
    pretrained_model = load_model_with_sd(args.model, pretrained_sd, device)
    pretrained_results = evaluate_unlearning(pretrained_model, loaders, device, label='Pretrained')
    del pretrained_model

    # Discover finetuned checkpoints
    ckpt_paths = sorted(glob.glob(os.path.join(args.finetuned_dir, '*.pt')))
    # Exclude any non-finetuned files
    ckpt_paths = [p for p in ckpt_paths if 'pretrained' not in os.path.basename(p)]
    print(f"\nFound {len(ckpt_paths)} finetuned checkpoints in {args.finetuned_dir}")
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No finetuned checkpoints in {args.finetuned_dir}. "
            "Run finetune_forget.py first."
        )

    # Build task vectors
    print("\n=== Building task vectors ===")
    task_vectors = []
    for ckpt_path in ckpt_paths:
        ft_sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        tv = ClassifierTaskVector(pretrained_sd=pretrained_sd, finetuned_sd=ft_sd)
        task_vectors.append(tv)
        print(f"  {os.path.basename(ckpt_path)}")

    # NegMerge: sign-consensus + negate
    print("\n=== NegMerge: sign-consensus merging ===")
    negmerge_tv = negmerge(task_vectors)  # already negated (for unlearning)

    # Coefficient sweep
    print(f"\n=== Coefficient sweep (n={args.n_eval_points}) ===")
    sweep = sweep_coefficients(
        negmerge_tv, pretrained_sd, args.model, loaders, device,
        args.n_eval_points, args.retain_threshold
    )

    best_alpha = find_best_alpha(
        sweep, pretrained_results['retain_acc'], args.retain_threshold
    )
    print(f"\nBest alpha: {best_alpha:.4f}")

    # Final evaluation at best alpha
    print("\n=== Final evaluation at optimal alpha ===")
    best_sd = negmerge_tv.apply_to(pretrained_sd, scaling_coef=best_alpha)
    best_model = load_model_with_sd(args.model, best_sd, device)
    final_results = evaluate_unlearning(best_model, loaders, device, label='NegMerge')

    # Save
    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, f'{args.model}_negmerge_unlearned.pt')
    torch.save(best_sd, save_path)

    out = {
        'method': 'NegMerge',
        'model': args.model,
        'n_task_vectors': len(task_vectors),
        'best_alpha': best_alpha,
        'retain_threshold': args.retain_threshold,
        'pretrained': pretrained_results,
        'unlearned': final_results,
        'coefficient_sweep': sweep,
    }

    print("\n" + "=" * 70)
    print("NegMerge Results")
    print("=" * 70)
    print(f"  {'':25s}  {'Pretrained':>12}  {'Unlearned':>12}")
    print(f"  {'Forget accuracy':25s}  {pretrained_results['forget_acc']:>12.4f}  "
          f"{final_results['forget_acc']:>12.4f}  ← want ~{1/10:.2f}")
    print(f"  {'Retain accuracy':25s}  {pretrained_results['retain_acc']:>12.4f}  "
          f"{final_results['retain_acc']:>12.4f}  ← want to stay high")
    print(f"  {'Test accuracy':25s}  {pretrained_results['test_acc']:>12.4f}  "
          f"{final_results['test_acc']:>12.4f}")
    print(f"  {'MIA AUC':25s}  {pretrained_results['mia_auc']:>12.4f}  "
          f"{final_results['mia_auc']:>12.4f}  ← want ~0.5")
    print(f"\n  Optimal alpha:        {best_alpha:.4f}")
    print(f"  Num models merged:    {len(task_vectors)}")
    print("=" * 70)

    results_file = os.path.join(args.results_dir, f'{args.model}_negmerge_results.json')
    with open(results_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_file}")
    print(f"Unlearned model saved to {save_path}")


if __name__ == '__main__':
    main()
