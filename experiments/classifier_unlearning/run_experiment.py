"""
Full CIFAR-10 classifier unlearning experiment — orchestration script.

Runs the complete pipeline in sequence:
  1. Train base model on full CIFAR-10
  2. Fine-tune a pool of models on the forget set (varying LR/WD)
  3. Apply NegMerge and evaluate
  4. Apply Task Arithmetic baseline and evaluate
  5. (Optional) Retrain oracle on retain set only
  6. Print comparison table

All intermediate checkpoints are saved so individual steps can be re-run.

Usage (from repo root, with negmerge-env activated):
  # Full experiment with ResNet18 (takes ~30-60 min on GPU, longer on CPU/MPS)
  python experiments/classifier_unlearning/run_experiment.py --model resnet18

  # Quick test with fewer models and epochs:
  python experiments/classifier_unlearning/run_experiment.py \\
      --model resnet18 --train-epochs 5 --finetune-epochs 5 --n-finetune-models 6

  # Skip training if checkpoints already exist:
  python experiments/classifier_unlearning/run_experiment.py \\
      --model resnet18 --skip-train --skip-finetune

  # Include oracle (expensive — trains a full model from scratch on retain set):
  python experiments/classifier_unlearning/run_experiment.py --model resnet18 --oracle
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser('Full CIFAR-10 Unlearning Experiment')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--save-dir', type=str, default='checkpoints/classifier')
    parser.add_argument('--results-dir', type=str, default='results/classifier')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--forget-fraction', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=4)
    # Training
    parser.add_argument('--train-epochs', type=int, default=200)
    parser.add_argument('--train-lr', type=float, default=0.1)
    parser.add_argument('--train-wd', type=float, default=5e-4)
    # Fine-tuning pool
    parser.add_argument('--finetune-epochs', type=int, default=30)
    parser.add_argument('--n-finetune-models', type=int, default=30,
                        help='Subset of the 30-model grid (1–30). '
                             'Use 6 for a quick 2LR×3WD test.')
    # Evaluation
    parser.add_argument('--n-eval-points', type=int, default=21)
    parser.add_argument('--retain-threshold', type=float, default=0.95)
    # Flags
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip step 1 if pretrained_best.pt already exists')
    parser.add_argument('--skip-finetune', action='store_true',
                        help='Skip step 2 if finetuned checkpoints already exist')
    parser.add_argument('--oracle', action='store_true',
                        help='Also run retrain oracle (expensive)')
    return parser.parse_args()


def run(cmd, description):
    """Run a subprocess command and raise on failure."""
    print(f"\n{'=' * 70}")
    print(f"  {description}")
    print(f"{'=' * 70}")
    print(f"  $ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=True)
    return result


def build_lr_wd_flags(n_models):
    """Return --lr-values and --wd-values flags covering up to n_models combos."""
    from experiments.classifier_unlearning.finetune_forget import LR_GRID, WD_GRID
    grid = [(lr, wd) for lr in LR_GRID for wd in WD_GRID]
    grid = grid[:n_models]
    used_lrs = sorted(set(lr for lr, _ in grid))
    used_wds = sorted(set(wd for _, wd in grid))
    return (
        ['--lr-values', ','.join(str(x) for x in used_lrs)],
        ['--wd-values', ','.join(str(x) for x in used_wds)],
    )


def main():
    args = parse_args()

    model_dir   = os.path.join(args.save_dir, args.model)
    ft_dir      = os.path.join(model_dir, 'finetuned')
    pretrained  = os.path.join(model_dir, 'pretrained_best.pt')
    py = sys.executable

    common = [
        '--model',            args.model,
        '--data-dir',         args.data_dir,
        '--seed',             str(args.seed),
        '--batch-size',       str(args.batch_size),
        '--num-workers',      str(args.num_workers),
        '--forget-fraction',  str(args.forget_fraction),
    ]

    # ── Step 1: Train base model ──────────────────────────────────────────
    if not args.skip_train or not os.path.exists(pretrained):
        run([py, 'experiments/classifier_unlearning/train.py'] + common + [
            '--epochs',    str(args.train_epochs),
            '--lr',        str(args.train_lr),
            '--wd',        str(args.train_wd),
            '--save-dir',  args.save_dir,
        ], f"Step 1: Train {args.model} on full CIFAR-10 ({args.train_epochs} epochs)")
    else:
        print(f"\nStep 1 skipped — found {pretrained}")

    # ── Step 2: Fine-tune pool on forget set ──────────────────────────────
    lr_flags, wd_flags = build_lr_wd_flags(args.n_finetune_models)
    if not args.skip_finetune:
        run([py, 'experiments/classifier_unlearning/finetune_forget.py'] + common[:-2] + [
            '--pretrained-path', pretrained,
            '--split-dir',       model_dir,
            '--save-dir',        ft_dir,
            '--epochs',          str(args.finetune_epochs),
        ] + lr_flags + wd_flags, f"Step 2: Fine-tune {args.n_finetune_models} models on forget set")
    else:
        print(f"\nStep 2 skipped — using existing checkpoints in {ft_dir}")

    eval_common = [
        '--model',              args.model,
        '--pretrained-path',    pretrained,
        '--finetuned-dir',      ft_dir,
        '--split-dir',          model_dir,
        '--results-dir',        args.results_dir,
        '--data-dir',           args.data_dir,
        '--batch-size',         str(args.batch_size),
        '--seed',               str(args.seed),
        '--num-workers',        str(args.num_workers),
        '--n-eval-points',      str(args.n_eval_points),
        '--retain-threshold',   str(args.retain_threshold),
    ]

    # ── Step 3: NegMerge ─────────────────────────────────────────────────
    run([py, 'experiments/classifier_unlearning/run_negmerge.py'] + eval_common,
        "Step 3: NegMerge evaluation")

    # ── Step 4: Task Arithmetic baseline ─────────────────────────────────
    run([py, 'experiments/classifier_unlearning/run_baseline.py'] + eval_common,
        "Step 4: Task Arithmetic baseline")

    # ── Step 5 (optional): Retrain oracle ────────────────────────────────
    if args.oracle:
        run([py, 'experiments/classifier_unlearning/retrain_oracle.py',
             '--model',       args.model,
             '--split-dir',   model_dir,
             '--results-dir', args.results_dir,
             '--save-dir',    model_dir,
             '--epochs',      str(args.train_epochs),
             '--data-dir',    args.data_dir,
             '--batch-size',  str(args.batch_size),
             '--num-workers', str(args.num_workers),
             '--seed',        str(args.seed),
             ], "Step 5: Retrain oracle on retain set")

    # ── Step 6: Comparison table ─────────────────────────────────────────
    run([py, 'experiments/classifier_unlearning/compare_results.py',
         '--model', args.model, '--results-dir', args.results_dir],
        "Step 6: Comparison table")


if __name__ == '__main__':
    main()
