"""
Train a CIFAR-10 classifier from scratch.

This produces the "pretrained" checkpoint that NegMerge and Task Arithmetic
will use as their base model. Saves the best checkpoint (by test accuracy)
and also records the forget/retain split indices for reproducibility.

Typical accuracy targets (no pretrained ImageNet weights):
  ResNet18:  ~94%
  VGG16:     ~93%
  Swin-T:    ~88% (harder to train on 32x32 without pretraining)

Usage (from repo root, with negmerge-env activated):
  python experiments/classifier_unlearning/train.py --model resnet18
  python experiments/classifier_unlearning/train.py --model vgg16
  python experiments/classifier_unlearning/train.py --model swin_t
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
    get_cifar10_loaders, get_model, accuracy
)


def parse_args():
    parser = argparse.ArgumentParser('Train CIFAR-10 classifier')
    parser.add_argument('--model', choices=['resnet18', 'vgg16', 'swin_t'], default='resnet18')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Total training epochs (200 gives good convergence for ResNet18)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('--save-dir', type=str, default='checkpoints/classifier')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--forget-fraction', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(images)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Device:  {device}")
    print(f"Model:   {args.model}")
    print(f"Epochs:  {args.epochs}  LR={args.lr}  WD={args.wd}")

    data_dir = os.path.expanduser(args.data_dir)
    loaders = get_cifar10_loaders(
        data_dir=data_dir,
        forget_fraction=args.forget_fraction,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"\nForget set size: {len(loaders['forget_indices'])} "
          f"({args.forget_fraction*100:.0f}% of train)")
    print(f"Retain set size: {len(loaders['retain_indices'])}")

    model = get_model(args.model, num_classes=10).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)

    save_dir = os.path.join(os.path.abspath(args.save_dir), args.model)
    os.makedirs(save_dir, exist_ok=True)

    best_acc, best_epoch = 0.0, 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders['full_train'], optimizer, criterion, device
        )
        scheduler.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            test_acc = accuracy(model, loaders['test'], device)
            forget_acc = accuracy(model, loaders['forget_eval'], device)
            retain_acc = accuracy(model, loaders['retain_eval'], device)

            row = {
                'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
                'test_acc': test_acc, 'forget_acc': forget_acc, 'retain_acc': retain_acc,
                'lr': scheduler.get_last_lr()[0],
            }
            history.append(row)
            print(f"Epoch {epoch:3d}/{args.epochs}  "
                  f"loss={train_loss:.4f}  train={train_acc:.4f}  "
                  f"test={test_acc:.4f}  forget={forget_acc:.4f}  retain={retain_acc:.4f}")

            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(
                    model.state_dict(),
                    os.path.join(save_dir, 'pretrained_best.pt')
                )

    # Save final checkpoint (used as starting point for finetuning)
    torch.save(model.state_dict(), os.path.join(save_dir, 'pretrained_final.pt'))

    # Save split indices so fine-tuning uses the same forget/retain partition
    np.save(os.path.join(save_dir, 'forget_indices.npy'), loaders['forget_indices'])
    np.save(os.path.join(save_dir, 'retain_indices.npy'), loaders['retain_indices'])

    # Save training metadata
    meta = {
        'model': args.model,
        'epochs': args.epochs,
        'lr': args.lr,
        'wd': args.wd,
        'seed': args.seed,
        'forget_fraction': args.forget_fraction,
        'best_test_acc': best_acc,
        'best_epoch': best_epoch,
        'n_forget': len(loaders['forget_indices']),
        'n_retain': len(loaders['retain_indices']),
    }
    with open(os.path.join(save_dir, 'train_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest test accuracy: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Checkpoints saved to: {save_dir}/")
    print("  pretrained_best.pt  ← use this for finetuning")
    print("  pretrained_final.pt ← last epoch")
    print("  forget_indices.npy  ← fixed forget/retain split")


if __name__ == '__main__':
    main()
