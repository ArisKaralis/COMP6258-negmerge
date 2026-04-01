"""
CLIP NegMerge: Cars unlearning using pre-computed checkpoints.

Replicates the CLIP zero-shot unlearning experiment from the NegMerge paper.
Uses the 30 pre-computed fine-tuned checkpoints in NegMerge_Checkpoints/
(each trained with a different RandAugment policy: m1-m10 x n1-n3).

Algorithm (from the paper):
  1. Compute task vectors: finetuned_i - pretrained
  2. Build sign-consensus mask: retain only params unanimous in sign across all models
  3. Merge masked vectors and negate to produce the unlearning direction
  4. Search for optimal scaling coefficient (min Cars acc, ImageNet >= 95% of pretrained)
  5. Evaluate on test set

Usage (run from repo root, with negmerge-env activated):
  export PYTHONPATH="$PYTHONPATH:$PWD/CLIP_MU"
  python experiments/clip_cars/run_negmerge.py \\
      --data-location ~/data \\
      --checkpoint-dir NegMerge_Checkpoints \\
      --results-dir results/clip_cars

  # If ImageNet is not available locally:
  python experiments/clip_cars/run_negmerge.py --skip-imagenet
"""

import sys
import os

# Ensure CLIP_MU src is on the path (works whether run from repo root or here)
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_repo_root, 'CLIP_MU'))

import argparse
import gc
import glob
import json
import numpy as np
import torch

from src.task_vectors import NonLinearTaskVector
from src.eval import eval_single_dataset
from src.utils import DotDict, find_optimal_coef


def parse_args():
    parser = argparse.ArgumentParser(description='NegMerge CLIP Cars Unlearning')
    parser.add_argument('--data-location', type=str,
                        default=os.path.expanduser('~/data'),
                        help='Root directory for datasets (Cars loaded from HuggingFace, '
                             'ImageNet expected at <data-location>/imagenet/val)')
    parser.add_argument('--checkpoint-dir', type=str, default='NegMerge_Checkpoints',
                        help='Directory with zeroshot.pt and finetuned checkpoints')
    parser.add_argument('--results-dir', type=str, default='results/clip_cars')
    parser.add_argument('--model', type=str, default='ViT-B-32')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-eval-points', type=int, default=21,
                        help='Number of coefficient values to sweep over [0, 1]')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--skip-imagenet', action='store_true',
                        help='Skip ImageNet evaluation entirely')
    parser.add_argument('--imagenet-data-location', type=str,
                        default=os.path.join(_repo_root, 'CLIP_MU', 'src', 'datasets_local'),
                        help='Parent dir containing imagenet/val/ for the ImageNetValSubset '
                             'control metric (default: CLIP_MU/src/datasets_local)')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def build_eval_args(args, checkpoint_dir, device):
    # Cars loads from HuggingFace and ignores data_location.
    # ImageNetValSubset needs data_location to point to the parent of imagenet/val/.
    data_location = (
        args.imagenet_data_location
        if not args.skip_imagenet
        else args.data_location
    )
    return DotDict({
        'device': device,
        'data_location': data_location,
        'batch_size': args.batch_size,
        'num_workers': getattr(args, 'num_workers', 4),
        'save': checkpoint_dir,           # heads are searched here first
        'results_db': checkpoint_dir,
        'model': args.model,
        'finetuning_mode': 'standard',
        'n_eval_points': args.n_eval_points,
        'openclip_cachedir': os.path.expanduser('~/openclip-cachedir/open_clip'),
        'control_dataset': None,
        'auto_aug': None,
        'cache_dir': None,
    })


def _clear_cache(device):
    gc.collect()
    if device == 'cuda' or (hasattr(device, 'type') and device.type == 'cuda'):
        torch.cuda.empty_cache()
    elif device == 'mps' or (hasattr(device, 'type') and device.type == 'mps'):
        torch.mps.empty_cache()


def _apply_tv_to_sd(pretrained_sd: dict, tv_vector: dict, scaling_coef: float) -> dict:
    """
    Apply a task vector to a state dict entirely on CPU.
    Returns a new state dict without touching the pretrained_sd or loading from disk.
    This avoids the repeated torch.load() calls that cause memory buildup in
    the original evaluate_task_vector() from eval.py.
    """
    new_sd = {}
    for key, val in pretrained_sd.items():
        if key in tv_vector:
            new_sd[key] = val + scaling_coef * tv_vector[key]
        else:
            new_sd[key] = val
    return new_sd


def sweep_clip_coefficients(
    tv_vector: dict,      # negated merged task vector dict (all on CPU)
    pretrained_model,     # ImageEncoder, kept on CPU between steps
    pretrained_sd: dict,  # pretrained state dict on CPU (never modified)
    datasets: list,       # e.g. ['CarsVal', 'ImageNetVal']
    eval_args,
    n_eval_points: int,
):
    """
    Memory-safe coefficient sweep for CLIP models.

    Keeps ONE ImageEncoder in memory throughout. For each alpha:
      1. Compute new state dict on CPU (never touches disk again).
      2. Load into the existing model object in-place.
      3. Move model to device, evaluate, move back to CPU.
      4. Free the intermediate state dict and flush device cache.

    This guarantees at most one model on the device at any point.

    Returns: dict mapping alpha → {dataset:top1, ...}
    """
    device = eval_args.device
    alphas = np.linspace(0.0, 1.0, n_eval_points)
    results = {}

    for alpha in alphas:
        # Compute new weights on CPU only
        new_sd = _apply_tv_to_sd(pretrained_sd, tv_vector, scaling_coef=float(alpha))

        # Load into existing model in-place, then free the dict
        pretrained_model.load_state_dict(new_sd)
        del new_sd
        gc.collect()

        # Move to device → evaluate → move back to CPU
        pretrained_model.to(device)
        per_dataset = {}
        for ds in datasets:
            metrics = eval_single_dataset(pretrained_model, ds, eval_args)
            per_dataset[f'{ds}:top1'] = metrics['top1']
        pretrained_model.to('cpu')
        _clear_cache(device)

        results[alpha] = per_dataset
        parts = '  '.join(f'{k}={v:.4f}' for k, v in per_dataset.items())
        print(f"  alpha={alpha:.2f}  {parts}")

    # Restore original weights so the model is left in a clean state
    pretrained_model.load_state_dict(pretrained_sd)
    return results


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    pretrained_checkpoint = os.path.join(checkpoint_dir, 'zeroshot.pt')

    assert os.path.exists(pretrained_checkpoint), \
        f"Pretrained checkpoint not found: {pretrained_checkpoint}"

    with open(os.path.join(checkpoint_dir, 'zeroshot_accuracies.json')) as f:
        pretrained_accuracies = json.load(f)

    print(f"Pretrained Cars accuracy:    {pretrained_accuracies['Cars']:.4f}")
    print(f"Pretrained CarsVal accuracy: {pretrained_accuracies['CarsVal']:.4f}")

    # --- Discover all finetuned Cars checkpoints ---------------------
    pattern = os.path.join(checkpoint_dir, 'clip-vit-b-32_cars_*.pt')
    checkpoint_paths = sorted(glob.glob(pattern))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No Cars finetuned checkpoints found at: {pattern}")
    print(f"\nFound {len(checkpoint_paths)} finetuned Cars checkpoints")

    # --- NegMerge: sign-consensus task vector merging ----------------
    print("\n=== Building sign-consensus task vector ===")
    merged_vector = None
    mask = None
    n = len(checkpoint_paths)

    for idx, ckpt_path in enumerate(checkpoint_paths):
        aug_tag = os.path.basename(ckpt_path).replace('clip-vit-b-32_cars_', '').replace('_finetuned.pt', '')
        print(f"  [{idx+1:02d}/{n}] {aug_tag}")
        tv = NonLinearTaskVector(pretrained_checkpoint, ckpt_path)

        if merged_vector is None:
            merged_vector = {k: torch.zeros_like(v) for k, v in tv.vector.items()}
            mask = {k: torch.zeros_like(v) for k, v in tv.vector.items()}

        for key in tv.vector:
            merged_vector[key] = merged_vector[key] + tv.vector[key]
            mask[key] = mask[key] + torch.sign(tv.vector[key])

        del tv  # free this checkpoint's task vector before loading the next

    # Apply sign-consensus: zero out parameters without unanimous sign
    n_total, n_kept = 0, 0
    for key in merged_vector:
        consensus = torch.abs(mask[key]) == n
        merged_vector[key] = torch.where(
            consensus,
            merged_vector[key] / n,
            torch.zeros_like(merged_vector[key])
        )
        n_total += consensus.numel()
        n_kept += int(consensus.sum().item())

    sparsity = 1.0 - n_kept / n_total
    print(f"\nSign-consensus sparsity: {100 * sparsity:.1f}% zeroed out  "
          f"({n_kept:,} / {n_total:,} parameters kept)")

    # The negated merged vector (all tensors on CPU)
    negated_vector = {k: -v for k, v in merged_vector.items()}
    del merged_vector, mask  # free now — only need negated_vector going forward
    gc.collect()

    # --- Load pretrained model ONCE for the entire sweep -------------
    # We do this here (not inside the loop) so it is never re-loaded from disk.
    print("\nLoading pretrained model once for coefficient sweep...")
    pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    eval_args = build_eval_args(args, checkpoint_dir, device)

    # --- Compute pretrained ImageNetValSubset accuracy (live, not from json) ---
    imagenet_pretrained_acc = None
    if not args.skip_imagenet:
        print("\nComputing pretrained ImageNetValSubset accuracy for control threshold...")
        pretrained_model.to(device)
        imagenet_pretrained_acc = eval_single_dataset(
            pretrained_model, 'ImageNetValSubset', eval_args
        )['top1']
        pretrained_model.to('cpu')
        _clear_cache(device)
        print(f"Pretrained ImageNetValSubset accuracy: {imagenet_pretrained_acc:.4f}")

    # --- Coefficient search on validation set (memory-safe) ----------
    val_datasets = ['CarsVal']
    if not args.skip_imagenet:
        val_datasets.append('ImageNetValSubset')

    print("\n=== Coefficient search on validation set ===")
    val_metrics = sweep_clip_coefficients(
        negated_vector, pretrained_model, pretrained_sd,
        val_datasets, eval_args, args.n_eval_points,
    )

    control_threshold = None
    if not args.skip_imagenet:
        control_threshold = 0.95 * imagenet_pretrained_acc
        print(f"\nImageNet control threshold (95% of pretrained): {control_threshold:.4f}")

    optimal_coef = find_optimal_coef(
        val_metrics,
        metric='CarsVal:top1',
        minimize=True,
        control_metric='ImageNetValSubset:top1' if not args.skip_imagenet else None,
        control_metric_threshold=control_threshold or 0.0,
    )

    if optimal_coef is None:
        print("WARNING: No coefficient met control threshold. Falling back to coef=1.0.")
        optimal_coef = 1.0

    val_cars = val_metrics[optimal_coef]['CarsVal:top1']
    print(f"Optimal coef: {optimal_coef:.4f}  (val Cars acc: {val_cars:.4f})")

    # --- Final evaluation on test set (reuses same model) ------------
    test_datasets = ['Cars']
    if not args.skip_imagenet:
        test_datasets.append('ImageNetValSubset')

    print("\n=== Test set evaluation ===")
    new_sd = _apply_tv_to_sd(pretrained_sd, negated_vector, scaling_coef=float(optimal_coef))
    pretrained_model.load_state_dict(new_sd)
    del new_sd
    pretrained_model.to(device)
    test_metrics = {}
    for ds in test_datasets:
        metrics = eval_single_dataset(pretrained_model, ds, eval_args)
        test_metrics[f'{ds}:top1'] = metrics['top1']
    pretrained_model.to('cpu')
    _clear_cache(device)

    # Save the unlearned encoder
    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, 'negmerge_unlearned_encoder.pt')
    torch.save(pretrained_model.state_dict(), save_path)

    # --- Print and save results --------------------------------------
    results = {
        'method': 'NegMerge',
        'num_checkpoints': n,
        'sparsity': sparsity,
        'n_params_kept': n_kept,
        'n_params_total': n_total,
        'optimal_coef': float(optimal_coef),
        'pretrained_cars_acc': pretrained_accuracies['Cars'],
        'unlearned_cars_acc': test_metrics.get('Cars:top1'),
        'pretrained_imagenet_acc': imagenet_pretrained_acc,
        'unlearned_imagenet_acc': test_metrics.get('ImageNetValSubset:top1'),
        'val_metrics_at_optimal': val_metrics[optimal_coef],
    }

    print("\n" + "=" * 70)
    print("NegMerge Results")
    print("=" * 70)
    print(f"  Pretrained Cars:     {results['pretrained_cars_acc']:.4f}")
    print(f"  Unlearned Cars:      {results['unlearned_cars_acc']:.4f}  ← should approach chance")
    if results['unlearned_imagenet_acc'] is not None:
        print(f"  Pretrained ImageNetValSubset: {results['pretrained_imagenet_acc']:.4f}")
        print(f"  Unlearned ImageNetValSubset:  {results['unlearned_imagenet_acc']:.4f}  ← should stay high")
    print(f"  Sparsity:            {100*sparsity:.1f}%")
    print(f"  Optimal coef:        {optimal_coef:.4f}")
    print("=" * 70)

    out_file = os.path.join(args.results_dir, 'negmerge_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    print(f"Unlearned encoder saved to {save_path}")


if __name__ == '__main__':
    main()
