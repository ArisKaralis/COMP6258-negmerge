"""
Task Arithmetic baseline for CLIP Cars unlearning.

For each of the 30 finetuned checkpoints, negates its task vector and finds the
optimal coefficient. Reports the best single-model result (lowest forget accuracy
while keeping ImageNet above 95% of pretrained). This is the standard Task
Arithmetic baseline that NegMerge is compared against.

Usage (from repo root, with negmerge-env activated):
  export PYTHONPATH="$PYTHONPATH:$PWD/CLIP_MU"
  python experiments/clip_cars/run_baseline.py \\
      --data-location ~/data \\
      --checkpoint-dir NegMerge_Checkpoints \\
      --results-dir results/clip_cars

  # Skip ImageNet if not available locally:
  python experiments/clip_cars/run_baseline.py --skip-imagenet
"""

import sys
import os

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


def _clear_cache(device):
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    elif device == 'mps':
        torch.mps.empty_cache()


def _apply_tv_to_sd(pretrained_sd: dict, tv_vector: dict, scaling_coef: float) -> dict:
    new_sd = {}
    for key, val in pretrained_sd.items():
        if key in tv_vector:
            new_sd[key] = val + scaling_coef * tv_vector[key]
        else:
            new_sd[key] = val
    return new_sd


def parse_args():
    parser = argparse.ArgumentParser(description='Task Arithmetic CLIP Cars Unlearning Baseline')
    parser.add_argument('--data-location', type=str, default=os.path.expanduser('~/data'))
    parser.add_argument('--checkpoint-dir', type=str, default='NegMerge_Checkpoints')
    parser.add_argument('--results-dir', type=str, default='results/clip_cars')
    parser.add_argument('--model', type=str, default='ViT-B-32')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--n-eval-points', type=int, default=21)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--skip-imagenet', action='store_true',
                        help='Skip ImageNet evaluation entirely')
    parser.add_argument('--imagenet-data-location', type=str,
                        default=os.path.join(_repo_root, 'CLIP_MU', 'src', 'datasets_local'),
                        help='Parent dir containing imagenet/val/ for ImageNetValSubset control')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def main():
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    pretrained_checkpoint = os.path.join(checkpoint_dir, 'zeroshot.pt')

    with open(os.path.join(checkpoint_dir, 'zeroshot_accuracies.json')) as f:
        pretrained_accuracies = json.load(f)

    pattern = os.path.join(checkpoint_dir, 'clip-vit-b-32_cars_*.pt')
    checkpoint_paths = sorted(glob.glob(pattern))
    print(f"Found {len(checkpoint_paths)} finetuned checkpoints to evaluate as baselines")

    # Cars loads from HuggingFace and ignores data_location.
    # ImageNetValSubset needs data_location pointing to the parent of imagenet/val/.
    data_location = (
        args.imagenet_data_location
        if not args.skip_imagenet
        else args.data_location
    )
    eval_args = DotDict({
        'device': device,
        'data_location': data_location,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'save': checkpoint_dir,
        'results_db': checkpoint_dir,
        'model': args.model,
        'finetuning_mode': 'standard',
        'n_eval_points': args.n_eval_points,
        'openclip_cachedir': os.path.expanduser('~/openclip-cachedir/open_clip'),
        'control_dataset': 'ImageNetValSubset' if not args.skip_imagenet else None,
        'auto_aug': None,
        'cache_dir': None,
    })

    # Compute pretrained ImageNetValSubset accuracy live (not from json — different split)
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

    control_threshold = (0.95 * imagenet_pretrained_acc) if not args.skip_imagenet else 0.0
    if not args.skip_imagenet:
        print(f"ImageNet control threshold (95% of pretrained): {control_threshold:.4f}")

    # Load pretrained model ONCE — reused across all 30 checkpoint sweeps.
    # Each sweep applies state dicts in-place; device memory is flushed between steps.
    print("\nLoading pretrained model once for all coefficient sweeps...")
    pretrained_model = torch.load(pretrained_checkpoint, map_location='cpu', weights_only=False)
    pretrained_sd = {k: v.clone() for k, v in pretrained_model.state_dict().items()}

    val_datasets = ['CarsVal'] + (['ImageNetValSubset'] if not args.skip_imagenet else [])
    alphas = np.linspace(0.0, 1.0, args.n_eval_points)

    # --- Evaluate each checkpoint as a Task Arithmetic candidate -----
    best_ckpt = None
    best_val_cars = float('inf')
    best_coef = None
    per_checkpoint_results = {}

    for ckpt_path in checkpoint_paths:
        aug_tag = os.path.basename(ckpt_path).replace('clip-vit-b-32_cars_', '').replace('_finetuned.pt', '')
        print(f"\n--- {aug_tag} ---")

        # Build negated task vector on CPU, then free the finetuned checkpoint
        tv = NonLinearTaskVector(pretrained_checkpoint, ckpt_path)
        neg_vector = {k: -v for k, v in tv.vector.items()}
        del tv  # free task vector tensors

        # Sweep alphas in-place on the shared pretrained_model
        best_local_coef = None
        best_local_cars = float('inf')
        for alpha in alphas:
            new_sd = _apply_tv_to_sd(pretrained_sd, neg_vector, scaling_coef=float(alpha))
            pretrained_model.load_state_dict(new_sd)
            del new_sd
            gc.collect()

            pretrained_model.to(device)
            cars_acc = eval_single_dataset(pretrained_model, 'CarsVal', eval_args)['top1']
            if not args.skip_imagenet:
                inet_acc = eval_single_dataset(pretrained_model, 'ImageNetValSubset', eval_args)['top1']
            else:
                inet_acc = None
            pretrained_model.to('cpu')
            _clear_cache(device)

            meets_control = (inet_acc is None) or (inet_acc >= control_threshold)
            if meets_control and cars_acc < best_local_cars:
                best_local_cars = cars_acc
                best_local_coef = float(alpha)

        # Restore pretrained weights after this checkpoint's sweep
        pretrained_model.load_state_dict(pretrained_sd)
        del neg_vector

        if best_local_coef is None:
            print(f"  No valid coefficient (ImageNet threshold not met). Skipping.")
            per_checkpoint_results[aug_tag] = {'coef': None, 'val_cars': None}
            continue

        print(f"  Best coef={best_local_coef:.4f}  CarsVal={best_local_cars:.4f}")
        per_checkpoint_results[aug_tag] = {'coef': best_local_coef, 'val_cars': float(best_local_cars)}

        if best_local_cars < best_val_cars:
            best_val_cars = best_local_cars
            best_ckpt = ckpt_path
            best_coef = best_local_coef

    if best_ckpt is None:
        print("\nERROR: No checkpoint produced a valid result. "
              "Try --skip-imagenet or lower --n-eval-points.")
        return

    best_tag = os.path.basename(best_ckpt).replace('clip-vit-b-32_cars_', '').replace('_finetuned.pt', '')
    print(f"\n=== Best checkpoint: {best_tag} ===")
    print(f"Val Cars accuracy: {best_val_cars:.4f}  at coef={best_coef:.4f}")

    # --- Final test set evaluation with best checkpoint --------------
    best_tv = NonLinearTaskVector(pretrained_checkpoint, best_ckpt)
    best_neg_vector = {k: -v for k, v in best_tv.vector.items()}
    del best_tv

    test_datasets = ['Cars'] + (['ImageNetValSubset'] if not args.skip_imagenet else [])
    new_sd = _apply_tv_to_sd(pretrained_sd, best_neg_vector, scaling_coef=float(best_coef))
    pretrained_model.load_state_dict(new_sd)
    del new_sd, best_neg_vector
    gc.collect()

    pretrained_model.to(device)
    test_metrics = {}
    for ds in test_datasets:
        test_metrics[f'{ds}:top1'] = eval_single_dataset(pretrained_model, ds, eval_args)['top1']
    pretrained_model.to('cpu')
    _clear_cache(device)

    os.makedirs(args.results_dir, exist_ok=True)
    save_path = os.path.join(args.results_dir, 'baseline_unlearned_encoder.pt')
    torch.save(pretrained_model.state_dict(), save_path)

    results = {
        'method': 'TaskArithmetic',
        'best_checkpoint': best_tag,
        'optimal_coef': float(best_coef),
        'pretrained_cars_acc': pretrained_accuracies['Cars'],
        'unlearned_cars_acc': test_metrics.get('Cars:top1'),
        'pretrained_imagenet_acc': imagenet_pretrained_acc,
        'unlearned_imagenet_acc': test_metrics.get('ImageNetValSubset:top1'),
        'per_checkpoint_val_results': per_checkpoint_results,
    }

    print("\n" + "=" * 70)
    print("Task Arithmetic Baseline Results")
    print("=" * 70)
    print(f"  Best checkpoint:     {best_tag}")
    print(f"  Pretrained Cars:     {results['pretrained_cars_acc']:.4f}")
    print(f"  Unlearned Cars:      {results['unlearned_cars_acc']:.4f}")
    if results['unlearned_imagenet_acc'] is not None:
        print(f"  Pretrained ImageNetValSubset: {results['pretrained_imagenet_acc']:.4f}")
        print(f"  Unlearned ImageNetValSubset:  {results['unlearned_imagenet_acc']:.4f}")
    print(f"  Optimal coef:        {best_coef:.4f}")
    print("=" * 70)

    out_file = os.path.join(args.results_dir, 'baseline_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
