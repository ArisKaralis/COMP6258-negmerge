"""
Fine-tune a pool of models on the forget set with varying hyperparameters.

This generates the candidate task vectors used by NegMerge. Each model starts
from the pretrained checkpoint and is fine-tuned for 30 epochs on the forget set
only, using AdamW with different (lr, wd) combinations.

The hyperparameter grid (10 LRs × 3 WDs = 30 models) mirrors the 30 CLIP
checkpoints in NegMerge_Checkpoints/.

Usage (from repo root, with negmerge-env activated):
  python experiments/classifier_unlearning/finetune_forget.py \\
      --model resnet18 \\
      --pretrained-path checkpoints/classifier/resnet18/pretrained_best.pt \\
      --split-dir checkpoints/classifier/resnet18 \\
      --save-dir checkpoints/classifier/resnet18/finetuned \\
      --epochs 30

Outputs: one .pt state-dict per (lr, wd) combination, plus a summary JSON.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.classifier_unlearning.utils import (
    get_cifar10_loaders, get_model, accuracy
)


# Hyperparameter grid (10 LRs × 3 WDs = 30 models)
LR_GRID = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2e-1]
WD_GRID  = [0.0, 1e-4, 1e-2]


def parse_args():
    parser = argparse.ArgumentParser('Fine-tune pool on forget set (NegMerge)')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--pretrained-path', type=str,
                        default='checkpoints/classifier/resnet18/pretrained_best.pt',
                        help='State dict of the pretrained model')
    parser.add_argument('--split-dir', type=str,
                        default='checkpoints/classifier/resnet18',
                        help='Directory with forget_indices.npy and retain_indices.npy')
    parser.add_argument('--save-dir', type=str,
                        default='checkpoints/classifier/resnet18/finetuned',
                        help='Where to save finetuned checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Fine-tuning epochs per model (paper uses 30)')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--forget-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    # Subset of grid (useful for quick tests)
    parser.add_argument('--lr-values', type=str, default=None,
                        help='Comma-separated LR values (overrides default grid)')
    parser.add_argument('--wd-values', type=str, default=None,
                        help='Comma-separated WD values (overrides default grid)')
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def finetune_one(model_init_sd, model_arch, forget_loader, device,
                 lr, wd, epochs):
    """Fine-tune a fresh copy of the model on the forget set."""
    model = get_model(model_arch, num_classes=10)
    model.load_state_dict(copy.deepcopy(model_init_sd))
    model = model.to(device)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    for epoch in range(1, epochs + 1):
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in forget_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        if epoch == epochs:
            final_loss = total_loss / total
            final_acc  = correct / total

    return model.state_dict(), final_loss, final_acc


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load pretrained state dict
    pretrained_sd = torch.load(args.pretrained_path, map_location='cpu', weights_only=True)
    print(f"Loaded pretrained model from {args.pretrained_path}")

    # Load the fixed forget/retain split
    forget_idx = np.load(os.path.join(args.split_dir, 'forget_indices.npy'))
    retain_idx  = np.load(os.path.join(args.split_dir, 'retain_indices.npy'))
    print(f"Forget set: {len(forget_idx)} samples  |  Retain set: {len(retain_idx)} samples")

    data_dir = os.path.expanduser(args.data_dir)
    loaders = get_cifar10_loaders(
        data_dir=data_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        forget_indices=forget_idx,
        retain_indices=retain_idx,
    )

    # Build hyperparameter grid
    lr_grid = [float(x) for x in args.lr_values.split(',')] if args.lr_values else LR_GRID
    wd_grid  = [float(x) for x in args.wd_values.split(',')] if args.wd_values else WD_GRID
    hp_grid = [(lr, wd) for lr in lr_grid for wd in wd_grid]
    print(f"\nHyperparameter grid: {len(lr_grid)} LRs × {len(wd_grid)} WDs = {len(hp_grid)} models")
    print(f"LR values: {lr_grid}")
    print(f"WD values: {wd_grid}")
    print(f"Epochs per model: {args.epochs}")

    os.makedirs(args.save_dir, exist_ok=True)

    results = []
    for i, (lr, wd) in enumerate(hp_grid):
        tag = f"lr{lr:.0e}_wd{wd:.0e}".replace('e-0', 'e-').replace('e+0', 'e')
        ckpt_path = os.path.join(args.save_dir, f'{tag}.pt')

        if os.path.exists(ckpt_path):
            print(f"[{i+1:02d}/{len(hp_grid)}] {tag} — already exists, skipping")
            results.append({'tag': tag, 'lr': lr, 'wd': wd, 'skipped': True})
            continue

        print(f"[{i+1:02d}/{len(hp_grid)}] {tag}  lr={lr}  wd={wd}")

        ft_sd, final_loss, final_acc = finetune_one(
            pretrained_sd, args.model,
            loaders['forget_train'], device,
            lr=lr, wd=wd, epochs=args.epochs,
        )

        torch.save(ft_sd, ckpt_path)

        row = {'tag': tag, 'lr': lr, 'wd': wd,
               'forget_train_loss': final_loss, 'forget_train_acc': final_acc,
               'ckpt_path': ckpt_path}
        results.append(row)
        print(f"  → forget_train_loss={final_loss:.4f}  forget_train_acc={final_acc:.4f}")

    summary = {
        'model': args.model,
        'pretrained_path': args.pretrained_path,
        'epochs': args.epochs,
        'n_models': len(results),
        'save_dir': args.save_dir,
        'checkpoints': results,
    }
    summary_path = os.path.join(args.save_dir, 'finetune_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. {len(results)} checkpoints saved to {args.save_dir}/")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
