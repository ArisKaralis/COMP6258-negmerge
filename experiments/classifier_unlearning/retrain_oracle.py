"""
Retrain oracle: train a model from scratch on the RETAIN set only.

This is the gold-standard for machine unlearning. A model that was never
exposed to the forget set should have:
  - Low forget accuracy (it never learned those samples)
  - High retain accuracy (trained normally on retain)

NegMerge and Task Arithmetic are considered good unlearning methods if
their results approach this oracle.

Usage (from repo root, with negmerge-env activated):
  python experiments/classifier_unlearning/retrain_oracle.py \\
      --model resnet18 \\
      --split-dir checkpoints/classifier/resnet18 \\
      --results-dir results/classifier
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from experiments.classifier_unlearning.utils import (
    get_cifar10_loaders, get_model, accuracy, evaluate_unlearning,
)


def parse_args():
    parser = argparse.ArgumentParser('Retrain Oracle on Retain Set')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--split-dir', type=str,
                        default='checkpoints/classifier/resnet18')
    parser.add_argument('--results-dir', type=str, default='results/classifier')
    parser.add_argument('--save-dir', type=str,
                        default='checkpoints/classifier/resnet18')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main():
    args = parse_args()
    device = get_device()
    torch.manual_seed(args.seed)

    forget_idx = np.load(os.path.join(args.split_dir, 'forget_indices.npy'))
    retain_idx  = np.load(os.path.join(args.split_dir, 'retain_indices.npy'))

    data_dir = os.path.expanduser(args.data_dir)
    loaders = get_cifar10_loaders(
        data_dir=data_dir, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
        forget_indices=forget_idx, retain_indices=retain_idx,
    )

    model = get_model(args.model, num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    print(f"Retraining {args.model} on retain set ({len(retain_idx)} samples)")
    print(f"Epochs: {args.epochs}  LR={args.lr}  WD={args.wd}")

    best_acc, best_sd = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for images, labels in loaders['retain_train']:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            test_acc = accuracy(model, loaders['test'], device)
            print(f"Epoch {epoch:3d}/{args.epochs}  test_acc={test_acc:.4f}")
            if test_acc > best_acc:
                best_acc = test_acc
                best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_sd)

    print("\n=== Retrain Oracle Evaluation ===")
    results = evaluate_unlearning(model, loaders, device, label='Oracle')

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir), exist_ok=True)

    save_path = os.path.join(args.save_dir, 'oracle_retrained.pt')
    torch.save(best_sd, save_path)

    out = {
        'method': 'RetrainOracle',
        'model': args.model,
        'epochs': args.epochs,
        **results
    }

    results_file = os.path.join(args.results_dir, f'{args.model}_oracle_results.json')
    with open(results_file, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"\nOracle forget_acc: {results['forget_acc']:.4f}  "
          f"(ideal target for NegMerge/TA)")
    print(f"Oracle retain_acc: {results['retain_acc']:.4f}")
    print(f"Oracle test_acc:   {results['test_acc']:.4f}")
    print(f"Oracle MIA AUC:    {results['mia_auc']:.4f}")
    print(f"\nSaved to {results_file}")


if __name__ == '__main__':
    main()
