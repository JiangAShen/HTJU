"""ImageNet Training with HTJU Optimizer (ResNet-50).

Paper: Table 8 — Hyperparameters for ResNet-50 on ImageNet

Usage:
    python train_imagenet.py --optimizer htju --data_dir /path/to/imagenet
    python train_imagenet.py --optimizer sgd --data_dir /path/to/imagenet
    python train_imagenet.py --optimizer adam --data_dir /path/to/imagenet
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import json
import time

from models.resnet import ResNet50
from htju import HTJU, CosineAnnealingLRDual


def get_args():
    p = argparse.ArgumentParser(description='ImageNet Training with HTJU')
    p.add_argument('--optimizer', default='htju', choices=['SGD', 'Adam', 'AdamW', 'htju'])
    p.add_argument('--lr', type=float, default=None)
    p.add_argument('--weight_decay', type=float, default=None)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--warmup_epochs', type=int, default=5)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to ImageNet directory containing train/ and val/')
    p.add_argument('--output_dir', type=str, default='./results_imagenet')
    return p.parse_args()


# Paper Table 8
HTJU_CONFIG = {
    't_lr': 1e-3,
    'h_lr': 1e-3,
    't_weight_decay': 1e-3,
    'h_weight_decay': 1e-4,
    'H_optim': 'SGD',
    'momentum': 0.9,
    't_betas': (0.9, 0.999),
}

BASELINE_DEFAULTS = {
    'SGD':    {'lr': 0.1,  'wd': 1e-4},
    'Adam':   {'lr': 1e-3, 'wd': 1e-4},
    'AdamW':  {'lr': 1e-3, 'wd': 1e-3},
}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = ResNet50(num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: ResNet-50 ({n_params:.1f}M params)")

    # Data
    from torchvision import datasets, transforms
    train_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'train'),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    val_dataset = datasets.ImageFolder(
        os.path.join(args.data_dir, 'val'),
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    criterion = nn.CrossEntropyLoss()

    # Optimizer
    opt_name = args.optimizer
    if opt_name == 'htju':
        cfg = HTJU_CONFIG
        optimizer = HTJU(
            model.parameters(),
            t_lr=cfg['t_lr'],
            h_lr=cfg['h_lr'],
            t_betas=cfg['t_betas'],
            momentum=cfg['momentum'],
            t_weight_decay=cfg['t_weight_decay'],
            h_weight_decay=cfg['h_weight_decay'],
            weight_decay_type='AdamW',
            H_optim=cfg['H_optim'],
            total_epoch=args.epochs,
            epoch_now=0,
        )
    else:
        bd = BASELINE_DEFAULTS[opt_name]
        lr = args.lr or bd['lr']
        wd = args.weight_decay if args.weight_decay is not None else bd['wd']
        if opt_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=wd)
        elif opt_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    if opt_name == 'htju':
        scheduler = CosineAnnealingLRDual(optimizer, T_max=args.epochs,
                                          t_lr_min=1e-6, h_lr_min=1e-6)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    print(f"\n{'='*70}")
    print(f"ResNet-50 | {opt_name} | ImageNet | {args.epochs}ep | bs={args.batch_size} | seed={args.seed}")
    print(f"{'='*70}\n")

    best_acc = 0.0
    best_acc5 = 0.0
    history = []
    t0 = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * data.size(0)
            train_total += data.size(0)

        # Advance HTJU epoch counter for s-weight schedule
        if opt_name == 'htju':
            optimizer.epoch_now += 1

        scheduler.step()
        train_loss = train_loss_sum / train_total
        epoch_time = time.time() - epoch_start

        # Evaluate
        model.eval()
        val_loss_sum = 0.0
        correct1, correct5, val_total = 0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss_sum += loss.item() * data.size(0)

                _, pred1 = output.topk(1, dim=1)
                correct1 += pred1.eq(target.view_as(pred1)).sum().item()
                _, pred5 = output.topk(5, dim=1)
                correct5 += pred5.eq(target.view_as(pred5)).sum().item()
                val_total += target.size(0)

        val_loss = val_loss_sum / val_total
        top1 = 100.0 * correct1 / val_total
        top5 = 100.0 * correct5 / val_total
        if top1 > best_acc:
            best_acc = top1
            best_acc5 = top5

        if opt_name == 'htju':
            t_lr_val = optimizer.param_groups[0]['t_lr']
            h_lr_val = optimizer.param_groups[0]['h_lr']
            cur_lr_str = f"t_lr={t_lr_val:.2e} h_lr={h_lr_val:.2e}"
        else:
            lr_val = optimizer.param_groups[0]['lr']
            cur_lr_str = f"lr={lr_val:.2e}"
        print(f"  ep {epoch+1:3d}/{args.epochs} | t_loss={train_loss:.4f} | "
              f"v_loss={val_loss:.4f} | top1={top1:.2f}% | top5={top5:.2f}% | "
              f"best={best_acc:.2f}% | {cur_lr_str} | {epoch_time:.0f}s")

        if opt_name == 'htju':
            history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                            'top1': top1, 'top5': top5,
                            't_lr': t_lr_val, 'h_lr': h_lr_val,
                            'epoch_time': epoch_time})
        else:
            history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss,
                            'top1': top1, 'top5': top5, 'lr': lr_val,
                            'epoch_time': epoch_time})

    total_time = time.time() - t0
    os.makedirs(args.output_dir, exist_ok=True)
    result = {
        'model': 'ResNet-50', 'optimizer': opt_name, 'epochs': args.epochs,
        'batch_size': args.batch_size, 'seed': args.seed,
        'best_top1': float(best_acc), 'best_top5': float(best_acc5),
        'total_time_h': total_time / 3600, 'history': history,
    }
    if opt_name == 'htju':
        result.update(HTJU_CONFIG)

    fname = f"ResNet50_{opt_name}_seed{args.seed}.json"
    with open(os.path.join(args.output_dir, fname), 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\n  => Best top-1: {best_acc:.2f}% top-5: {best_acc5:.2f}%")
    print(f"  => Saved to {os.path.join(args.output_dir, fname)}")
    print(f"  => Total time: {total_time/3600:.1f}h")


if __name__ == '__main__':
    main()
