# HTJU: Homotopy Trajectory-Unified Optimization

Official implementation of the HTJU optimizer family. Three adaptive optimizers featuring **approximate diagonal Hessian** preconditioning — from the foundational TJU to the homotopy-blended HTJU.

## Optimizers

| Class | Paper Name | File | Description |
|-------|-----------|------|-------------|
| `TJU` | TJU (Atom) | `htju/tju.py` | Single-beta momentum + L4-norm approximate diagonal Hessian via outer product of parameter updates |
| `NdaTJU` | NdaTJU | `htju/tju_v4.py` | TJU + Adam-style β₁/β₂ + AdamW decoupled weight decay + additive Hessian fusion + cosine annealing |
| `HTJU` | HTJU | `htju/htju.py` | NdaTJU + homotopy dual-optimizer: a second optimizer (SGD/Adam/AdamW) runs in parallel, blended via sigmoid s-weight |

### Ablation Pathway

```
TJU (Atom)
  │  + β₁/β₂ moments, decoupled AdamW, cosine scheduler
  ▼
NdaTJU
  │  + homotopy dual-optimizer, sigmoid s-weight blending
  ▼
HTJU
```

## Installation

```bash
pip install -e .
```

Requirements: Python >= 3.8, PyTorch >= 1.10

## Quick Start

```python
from htju import TJU, NdaTJU, HTJU, CosineAnnealingLRDual

model = YourModel()

# TJU — single-LR with approximate Hessian
opt = TJU(model.parameters(), lr=1e-3, beta=0.9, eps=1e-4)

# NdaTJU — AdamW decoupled weight decay + cosine scheduler
opt = NdaTJU(model.parameters(), lr=1e-3, betas=(0.9, 0.999),
             weight_decay=0.01, weight_decay_type='AdamW', hessian_scale=0.05)

# HTJU — homotopy dual-optimizer
opt = HTJU(model.parameters(), t_lr=1e-3, h_lr=1e-3,
           H_optim='AdamW', t_weight_decay=5e-4,
           weight_decay_type='AdamW', total_epoch=400, epoch_now=0,
           hessian_scale=0.05)

# HTJU requires epoch tracking for the s-weight schedule
for epoch in range(total_epochs):
    train(...)
    opt.epoch_now += 1

# Dual-LR cosine annealing for HTJU
scheduler = CosineAnnealingLRDual(opt, T_max=epochs, t_lr_min=1e-6, h_lr_min=1e-6)
```

## API Reference

### TJU
```python
TJU(params, lr=1e-3, beta=0.9, eps=1e-4, rebound='constant',
    warmup=500, init_lr=None, weight_decay=0, weight_decay_type='L2')
```

### NdaTJU (TJUv4)
```python
NdaTJU(params, lr=1e-3, betas=(0.9, 0.999), beta_h=0.85, eps=1e-8,
       rebound='constant', warmup=100, init_lr=None, weight_decay=0.0,
       weight_decay_type='AdamW', hessian_scale=0.05, total_steps=10000,
       use_cosine_scheduler=True)
```

### HTJU
```python
HTJU(params, t_lr=1e-3, h_lr=1e-3, t_betas=(0.9, 0.999),
     h_betas=(0.9, 0.999), beta_h=0.85, t_eps=0, h_eps=0,
     momentum=0, rebound='constant', warmup=0, init_lr=None,
     t_weight_decay=0.0, h_weight_decay=0, weight_decay_type='AdamW',
     H_optim=None, total_epoch=0, epoch_now=0, hessian_scale=0.05,
     total_steps=100000, use_cosine_scheduler=False)
```

### CosineAnnealingLRDual
```python
CosineAnnealingLRDual(optimizer, T_max, t_lr_min=0, h_lr_min=0, last_epoch=-1)
# Methods: step(), get_t_lr(), get_h_lr()
```

## Experiments

| Benchmark | Directory | Optimizers Compared |
|-----------|-----------|---------------------|
| CIFAR-10/100 | `experiments/classification/` | HTJU, SGD, Adam, AdamW |
| ImageNet | `experiments/classification/` | HTJU, SGD, Adam, AdamW |
| LSTM (PTB) | `experiments/nlp/lstm/` | HTJU, SGD, Adam, AdamW, RAdam, MSVAG, Yogi |
| Transformer-XL (WT103) | `experiments/nlp/transformer_xl/` | HTJU, Adam, SGD |
| U-KAN Segmentation | `experiments/segmentation/ukan/` | TJU, NdaTJU, HTJU, Adam, AdamW, SGD |
| PPO (MuJoCo) | `experiments/rl/ppo/` | HTJU, Adam, SGD |

U-KAN is the **ablation benchmark** comparing all three TJU-family optimizers. Other benchmarks use HTJU as the sole TJU-family method against standard baselines.

## Recommended Hyperparameters

| Parameter | TJU | NdaTJU | HTJU |
|-----------|-----|--------|------|
| Learning rate | lr=1e-3 | lr=1e-3 | t_lr=1e-3, h_lr=1e-3 |
| Betas | β=0.9 | (0.9, 0.999) | t_betas=(0.9, 0.999), h_betas=(0.9, 0.999) |
| Weight decay | 0 | 0.01 (AdamW) | t_wd=5e-4, h_wd=1e-4 (AdamW) |
| Hessian scale | — | 0.05 | 0.05 |
| Epsilon | 1e-4 | 1e-8 | t_eps=0, h_eps=0 |
| H_optim | — | — | AdamW |
| Warmup | 500 | 100 | 0 |
| Scheduler | — | CosineAnnealingLR | CosineAnnealingLRDual |

Task-specific configurations are documented in [configs/experiment_configs.md](configs/experiment_configs.md).

## Key Concepts

**Approximate Hessian.** TJU tracks a diagonal Hessian approximation via the outer product of parameter updates, providing curvature-aware preconditioning without second-order derivatives. Controlled by `hessian_scale` and `beta_h`.

**Homotopy Blending (HTJU).** A second optimizer (SGD/Adam/AdamW, configured via `H_optim`) runs in parallel. Their contributions are mixed via a sigmoid weight `s`:
```
s = 1 / (1 + exp(-18 * (epoch / total_epoch - 0.5)))
```
Early in training (`s ≈ 0`), the homotopy optimizer dominates for exploration. Later (`s ≈ 1`), the TJU branch takes over for fine convergence. The transition is controlled by incrementing `epoch_now` each epoch.

**Decoupled Weight Decay (NdaTJU, HTJU).** Following AdamW, weight decay is applied directly to parameters rather than mixed into the gradient. Controlled by `weight_decay_type='AdamW'`.

## Citation

If you use this code, please cite the HTJU paper.

## License

MIT — see [LICENSE](LICENSE)
