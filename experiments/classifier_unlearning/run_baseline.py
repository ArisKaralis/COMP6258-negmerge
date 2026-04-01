"""
Task Arithmetic baseline for classifier unlearning on CIFAR-10.

For each finetuned checkpoint, negates its task vector and sweeps the
coefficient. Picks the single model and coefficient that best minimises
forget accuracy while keeping retain accuracy above the threshold.

This is the standard Task Arithmetic negation baseline (one model, one
hyperparameter search), directly comparable to NegMerge.

Usage (from repo root, with negmerge-env activated):
  python experiments/classifier_unlearning/run_baseline.py \\
      --model resnet18 \\
      --pretrained-path checkpoints/classifier/resnet18/pretrained_best.pt \\
      --finetuned-dir checkpoints/classifier/resnet18/finetuned \\
      --split-dir checkpoints/classifier/resnet18 \\
      --results-dir results/classifier
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
    ClassifierTaskVector, accuracy, evaluate_unlearning,
)


def parse_args():
    parser = argparse.ArgumentParser('Task Arithmetic Baseline for Classifier Unlearning')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--pretrained-path', type=str,
                        default='checkpoints/classifier/resnet18/pretrained_best.pt')
    parser.add_argument('--finetuned-dir', type=str,
                        default='checkpoints/classifier/resnet18/finetuned')
    parser.add_argument('--split-dir', type=str,
                        default='checkpoints/classifier/resnet18')
    parser.add_argument('--results-dir', type=str, default='results/classifier')
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--n-eval-points', type=int, default=21)
    parser.add_argument('--retain-threshold', type=float, default=0.95)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _clear_cache(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def load_model_with_sd(model_arch, state_dict, device):
    model = get_model(model_arch, num_classes=10)
    model.load_state_dict(state_dict)
    return model.to(device)


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}\nModel:  {args.model}")

    pretrained_sd = torch.load(
        args.pretrained_path, map_location='cpu', weights_only=True
    )

    forget_idx = np.load(os.path.join(args.split_dir, 'forget_indices.npy'))
    retain_idx  = np.load(os.path.join(args.split_dir, 'retain_indices.npy'))

    data_dir = os.path.expanduser(args.data_dir)
    loaders = get_cifar10_loaders(
        data_dir=data_dir, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
        forget_indices=forget_idx, retain_indices=retain_idx,
    )

    # Pretrained baseline
    pretrained_model = load_model_with_sd(args.model, pretrained_sd, device)
    pretrained_retain = accuracy(pretrained_model, loaders['retain_eval'], device)
    print(f"\nPretrained retain accuracy: {pretrained_retain:.4f}")
    retain_threshold_abs = args.retain_threshold * pretrained_retain
    print(f"Retain threshold ({args.retain_threshold:.0%} of pretrained): {retain_threshold_abs:.4f}")
    del pretrained_model

    ckpt_paths = sorted(glob.glob(os.path.join(args.finetuned_dir, '*.pt')))
    ckpt_paths = [p for p in ckpt_paths if 'pretrained' not in os.path.basename(p)]
    print(f"\n{len(ckpt_paths)} finetuned checkpoints to evaluate")

    alphas = np.linspace(0.0, 1.0, args.n_eval_points)

    # For each checkpoint, find the best (alpha, checkpoint) pair
    best_result = None  # {'ckpt': ..., 'alpha': ..., 'forget_acc': ..., 'retain_acc': ...}
    all_per_ckpt = {}

    for ckpt_path in ckpt_paths:
        tag = os.path.basename(ckpt_path).replace('.pt', '')
        ft_sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        tv = ClassifierTaskVector(pretrained_sd=pretrained_sd, finetuned_sd=ft_sd)
        negated_tv = -tv

        print(f"\n  {tag}")
        best_local = None
        for alpha in alphas:
            # Compute state dict on CPU, free it before moving model to device
            sd = negated_tv.apply_to(pretrained_sd, scaling_coef=float(alpha))
            model = get_model(args.model, num_classes=10)
            model.load_state_dict(sd)
            del sd           # ← free CPU dict before device allocation
            model = model.to(device)

            forget_acc = accuracy(model, loaders['forget_eval'], device)
            retain_acc = accuracy(model, loaders['retain_eval'], device)
            del model
            _clear_cache(device)

            if retain_acc >= retain_threshold_abs:
                if best_local is None or forget_acc < best_local['forget_acc']:
                    best_local = {
                        'alpha': float(alpha),
                        'forget_acc': forget_acc,
                        'retain_acc': retain_acc,
                    }

        if best_local is None:
            print(f"    No alpha met retain threshold. Skipping.")
            all_per_ckpt[tag] = None
            continue

        print(f"    Best: alpha={best_local['alpha']:.2f}  "
              f"forget={best_local['forget_acc']:.4f}  "
              f"retain={best_local['retain_acc']:.4f}")
        all_per_ckpt[tag] = best_local

        if best_result is None or best_local['forget_acc'] < best_result['forget_acc']:
            best_result = {**best_local, 'ckpt': ckpt_path, 'tag': tag}

    if best_result is None:
        print("\nERROR: No valid (checkpoint, alpha) pair found. "
              "Try --retain-threshold 0.90 or run more fine-tuned models.")
        return

    print(f"\n=== Best checkpoint: {best_result['tag']} ===")
    print(f"Alpha: {best_result['alpha']:.4f}  "
          f"Forget: {best_result['forget_acc']:.4f}  "
          f"Retain: {best_result['retain_acc']:.4f}")

    # Full evaluation at best
    best_ft_sd = torch.load(best_result['ckpt'], map_location='cpu', weights_only=True)
    best_tv = ClassifierTaskVector(pretrained_sd=pretrained_sd, finetuned_sd=best_ft_sd)
    negated_best_tv = -best_tv
    final_sd = negated_best_tv.apply_to(pretrained_sd, scaling_coef=best_result['alpha'])
    final_model = load_model_with_sd(args.model, final_sd, device)

    # Pretrained full eval for comparison
    print("\n=== Pretrained model ===")
    pretrained_full_model = load_model_with_sd(args.model, pretrained_sd, device)
    pretrained_eval = evaluate_unlearning(
        pretrained_full_model, loaders, device, label='Pretrained'
    )
    del pretrained_full_model

    print("\n=== Task Arithmetic Baseline ===")
    final_eval = evaluate_unlearning(final_model, loaders, device, label='TA Baseline')

    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, f'{args.model}_baseline_unlearned.pt')
    torch.save(final_sd, save_path)

    out = {
        'method': 'TaskArithmetic',
        'model': args.model,
        'best_checkpoint': best_result['tag'],
        'best_alpha': best_result['alpha'],
        'retain_threshold': args.retain_threshold,
        'pretrained': pretrained_eval,
        'unlearned': final_eval,
        'per_checkpoint_results': {
            k: v for k, v in all_per_ckpt.items() if v is not None
        },
    }

    print("\n" + "=" * 70)
    print("Task Arithmetic Baseline Results")
    print("=" * 70)
    print(f"  {'':25s}  {'Pretrained':>12}  {'Unlearned':>12}")
    print(f"  {'Forget accuracy':25s}  {pretrained_eval['forget_acc']:>12.4f}  "
          f"{final_eval['forget_acc']:>12.4f}  ← want ~{1/10:.2f}")
    print(f"  {'Retain accuracy':25s}  {pretrained_eval['retain_acc']:>12.4f}  "
          f"{final_eval['retain_acc']:>12.4f}")
    print(f"  {'Test accuracy':25s}  {pretrained_eval['test_acc']:>12.4f}  "
          f"{final_eval['test_acc']:>12.4f}")
    print(f"  {'MIA AUC':25s}  {pretrained_eval['mia_auc']:>12.4f}  "
          f"{final_eval['mia_auc']:>12.4f}  ← want ~0.5")
    print(f"\n  Best checkpoint:   {best_result['tag']}")
    print(f"  Optimal alpha:     {best_result['alpha']:.4f}")
    print("=" * 70)

    results_file = os.path.join(args.results_dir, f'{args.model}_baseline_results.json')
    with open(results_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    main()
