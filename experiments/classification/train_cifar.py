"""CIFAR-10/100 Training with HTJU Optimizer.

Reference hyperparameters from the HTJU paper (Appendix C, Table 6 & 7).

Usage:
    # CIFAR-10 with HTJU (paper default)
    python train_cifar.py --dataset cifar10 --optimizer htju --epochs 100

    # CIFAR-100 with HTJU
    python train_cifar.py --dataset cifar100 --optimizer htju --epochs 200

    # Compare with baselines
    python train_cifar.py --dataset cifar100 --optimizer adam --epochs 200
    python train_cifar.py --dataset cifar100 --optimizer sgd --epochs 200
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.resnet import ResNet18, ResNet50
from htju import HTJU, CosineAnnealingLRDual


def get_args():
    p = argparse.ArgumentParser(description='CIFAR Training with HTJU')
    p.add_argument('--dataset', type=str, default='cifar100', choices=['cifar10', 'cifar100'])
    p.add_argument('--optimizer', type=str, default='htju',
                   choices=['sgd', 'adam', 'adamw', 'htju'])
    p.add_argument('--model', type=str, default=None,
                   help='ResNet model (auto-selected if not specified)')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', type=str, default='./data')
    p.add_argument('--lr', type=float, default=None, help='Override learning rate')
    return p.parse_args()


# Paper Table 6/7 hyperparameters
HTJU_CONFIGS = {
    'cifar10': {
        'model': 'resnet18',
        't_lr': 1e-3,
        'h_lr': 1e-2,
        't_weight_decay': 1e-1,  # Fixed: paper Table 6 says 1e-3(t)
        'h_weight_decay': 1e-3,
        'H_optim': 'SGD',
        'momentum': 0.9,
        't_betas': (0.9, 0.999),
        'h_betas': (0.9, 0.999),
        'hessian_scale': 0.05,
        't_lr_min': 1e-6,
        'h_lr_min': 1e-6,
        'epochs': 100,
    },
    'cifar100': {
        'model': 'resnet50',
        't_lr': 1e-3,
        'h_lr': 1e-2,
        't_weight_decay': 5e-1,  # Fixed: paper Table 7 says 1e-3(t)
        'h_weight_decay': 1e-3,
        'H_optim': 'SGD',
        'momentum': 0.9,
        't_betas': (0.9, 0.999),
        'h_betas': (0.9, 0.999),
        'hessian_scale': 0.05,
        't_lr_min': 1e-6,
        'h_lr_min': 1e-6,
        'epochs': 200,
    },
}

BASELINE_CONFIGS = {
    'sgd':      {'lr': 0.1,  'weight_decay': 5e-4, 'momentum': 0.9},
    'adam':     {'lr': 5e-3, 'weight_decay': 1e-4, 'betas': (0.9, 0.999)},
    'adamw':    {'lr': 1e-3, 'weight_decay': 1e-3, 'betas': (0.9, 0.999)},
}


def build_optimizer(model, args):
    cfg = HTJU_CONFIGS[args.dataset]

    if args.optimizer == 'htju':
        t_lr = args.lr or cfg['t_lr']
        optimizer = HTJU(
            model.parameters(),
            t_lr=t_lr,
            h_lr=cfg['h_lr'],
            t_betas=cfg['t_betas'],
            h_betas=cfg['h_betas'],
            momentum=cfg['momentum'],
            t_weight_decay=cfg['t_weight_decay'],
            h_weight_decay=cfg['h_weight_decay'],
            weight_decay_type='AdamW',
            H_optim=cfg['H_optim'],
            hessian_scale=cfg['hessian_scale'],
            total_epoch=args.epochs,
            epoch_now=0,
        )
        return optimizer
    else:
        bc = BASELINE_CONFIGS[args.optimizer]
        lr = args.lr or bc['lr']
        if args.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=bc['momentum'],
                             weight_decay=bc['weight_decay'])
        elif args.optimizer == 'adam':
            return optim.Adam(model.parameters(), lr=lr, weight_decay=bc['weight_decay'],
                              betas=bc['betas'])
        elif args.optimizer == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=bc['weight_decay'],
                               betas=bc['betas'])


def main():
    args = get_args()
    cfg = HTJU_CONFIGS[args.dataset]

    if args.epochs is None:
        args.epochs = cfg['epochs']

    num_classes = 10 if args.dataset == 'cifar10' else 100
    model_name = args.model or cfg['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Model (CIFAR-adapted: 3x3 conv + no maxpool for 32x32 images)
    if model_name == 'resnet18':
        model = ResNet18(num_classes=num_classes, cifar_mode=True)
    elif model_name == 'resnet50':
        model = ResNet50(num_classes=num_classes, cifar_mode=True)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {model_name} ({n_params:.1f}M params), Dataset: {args.dataset}")

    # Data
    mean = [0.5071, 0.4867, 0.4408] if args.dataset == 'cifar100' else [0.4914, 0.4822, 0.4465]
    std = [0.2675, 0.2565, 0.2761] if args.dataset == 'cifar100' else [0.2470, 0.2435, 0.2616]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    DatasetClass = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
    train_dataset = DatasetClass(args.data_dir, train=True, download=True, transform=transform_train)
    test_dataset = DatasetClass(args.data_dir, train=False, download=True, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    # Optimizer + Scheduler
    optimizer = build_optimizer(model, args)
    if args.optimizer == 'htju':
        # Dual-LR cosine annealing: anneals both t_lr and h_lr
        scheduler = CosineAnnealingLRDual(optimizer, T_max=args.epochs,
                                          t_lr_min=cfg['t_lr_min'],
                                          h_lr_min=cfg['h_lr_min'])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()

    print(f"\n{'='*70}")
    print(f"{model_name} | {args.optimizer} | {args.dataset} | "
          f"{args.epochs}ep | bs={args.batch_size} | seed={args.seed}")
    print(f"{'='*70}\n")

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            train_correct += output.argmax(1).eq(target).sum().item()
            train_total += target.size(0)

        # Update HTJU epoch for homotopy s-weight transition
        if args.optimizer == 'htju':
            optimizer.epoch_now += 1

        scheduler.step()
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                test_loss += loss.item() * data.size(0)
                test_correct += output.argmax(1).eq(target).sum().item()
                test_total += target.size(0)

        train_acc = 100.0 * train_correct / train_total
        test_acc = 100.0 * test_correct / test_total
        if test_acc > best_acc:
            best_acc = test_acc

        if args.optimizer == 'htju':
            cur_lr_str = f"t_lr={optimizer.param_groups[0]['t_lr']:.2e} h_lr={optimizer.param_groups[0]['h_lr']:.2e}"
        else:
            cur_lr_str = f"lr={optimizer.param_groups[0]['lr']:.2e}"
        print(f"  ep {epoch+1:3d}/{args.epochs} | t_loss={train_loss/train_total:.4f} | "
              f"t_acc={train_acc:.2f}% | v_loss={test_loss/test_total:.4f} | "
              f"v_acc={test_acc:.2f}% | best={best_acc:.2f}% | {cur_lr_str} | {epoch_time:.1f}s")

    total_time = time.time() - t0
    print(f"\n  => Best test accuracy: {best_acc:.2f}%")
    print(f"  => Total time: {total_time/60:.1f}min")


if __name__ == '__main__':
    print(torch.cuda.is_available())
    main()
