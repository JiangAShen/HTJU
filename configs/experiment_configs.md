# HTJU Experiment Configuration Reference

Hyperparameter configurations for reproducing paper results. All values from the HTJU paper.

---

## 1. NLP — LSTM Language Modeling (Penn Treebank)

**Model:** LSTM, embedding=400, hidden=1150, batch_size=20 (run scripts) / 80, BPTT=70, dropout=0.4

### Baseline Optimizer Configs

| Optimizer | lr (layer 1) | lr (layer 2-3) | eps | betas | weight_decay | clip |
|-----------|-------------|----------------|-----|-------|-------------|------|
| SGD | 30.0 | 30.0 | — | — | 1.2e-6 | 0.25 |
| Adam | 1e-3 | 1e-2 | 1e-12 / 1e-8 | (0.9, 0.999) | 1.2e-6 | 0.25 |
| AdamW | 1e-3 | 1e-2 | 1e-12 / 1e-8 | (0.9, 0.999) | 1.2e-6 | 0.25 |
| RAdam | 1e-3 | 1e-2 | 1e-12 / 1e-8 | (0.9, 0.999) | 1.2e-6 | 0.25 |
| MSVAG | 30.0 | 30.0 | 1e-8 | (0.9, 0.999) | 1.2e-6 | 0.25 |
| Yogi | 1e-2 | 1e-2 | 1e-3 | (0.9, 0.999) | 1.2e-6 | 0.25 |

LR divided by 10 at epochs [100, 145]. Epochs=200.

### HTJU Config

| Optimizer | t_lr | h_lr | t_weight_decay | H_optim | hessian_scale | dropout_i | dropout_h |
|-----------|------|------|---------------|---------|---------------|-----------|-----------|
| HTJU | 1e-3 (L1) / 1e-2 (L2-3) | same as t_lr | 1.2e-6 | AdamW | 0.05 | 0.45 | 0.3 |

**Settings:** `weight_decay_type='AdamW'`, `total_epoch=200`, `epoch_now=0`

---

## 2. NLP — Transformer-XL (WikiText-103)

**Model:** 16-layer, d_model=410, d_inner=2100, n_head=10, d_head=41, dropout=0.1, batch_size=60

### Baseline Configs

| Optimizer | lr | weight_decay | warmup_steps | max_step | clip |
|-----------|-----|-------------|-------------|----------|------|
| Adam | 2.5e-4 | 0.02 | 5000 | 200000 | 0.25 |
| SGD | 5.0 | 0.02 | 0 | 200000 | 0.25 |

### HTJU Config

| Optimizer | t_lr | h_lr | t_weight_decay | h_weight_decay | H_optim | hessian_scale |
|-----------|------|------|---------------|---------------|---------|---------------|
| HTJU | 2.5e-4 | 2.5e-4 | 0.02 | 0.02 | AdamW | 0.05 |

**Settings:** `weight_decay_type='AdamW'`, `total_epoch=max_step`, cosine LR with `CosineAnnealingLRDual`, `lr_min=1e-6`, gradient clipping=0.25

---

## 3. Medical Image Segmentation — U-KAN (Ablation)

**Model:** U-KAN, input=256x256, batch_size=8, epochs=400

### All Optimizer Configs

| Optimizer | lr / t_lr | h_lr | weight_decay | eps | betas | warmup | H_optim | hessian_scale |
|-----------|--------|------|-------------|-----|-------|--------|---------|---------------|
| Adam | 1e-4 | — | 5e-4 | — | — | — | — | — |
| AdamW | 1e-4 | — | 5e-4 | — | — | — | — | — |
| SGD | 1e-4 | — | 5e-4 | — | — | — | — | — |
| TJU | 1e-4 | — | 0 | 1e-6 | β=0.9 | 50 | — | — |
| NdaTJU | 5e-3 | — | 5e-4 | 1e-6 | (0.9, 0.999) | 50 | — | — |
| HTJU | 1e-3 | 1e-3 | t:5e-4 / h:1e-4 | t:1e-6 / h:1e-8 | (0.9, 0.999) | 0 | AdamW | 0.05 |

**KAN layer params (separate param group):** `kan_lr=1e-2`, `kan_weight_decay=1e-4`

**Settings:** `weight_decay_type='AdamW'` (NdaTJU/HTJU), `total_epoch=400`, `min_lr=1e-5`, `momentum=0.9`

---

## 4. Image Classification — CIFAR-10/100 (ResNet)

**Model:** ResNet-18 (CIFAR-10), ResNet-50 (CIFAR-100), batch_size=256, cosine LR

### HTJU Paper Configs (Table 6 & 7)

| Dataset | Model | t_lr | h_lr | t_weight_decay | h_weight_decay | H_optim | momentum | t_betas | epochs |
|---------|-------|------|------|---------------|---------------|---------|----------|---------|--------|
| CIFAR-10 | ResNet-18 | 5e-3 | 1e-3 | 1e-3 | 5e-4 | SGD | 0.9 | (0.9, 0.999) | 100 |
| CIFAR-100 | ResNet-50 | 5e-3 | 1e-3 | 1e-3 | 5e-4 | SGD | 0.9 | (0.9, 0.999) | 200 |

### Baseline Configs

| Optimizer | lr | weight_decay | momentum / betas |
|-----------|-----|-------------|-------------------|
| SGD | 0.1 | 5e-4 | momentum=0.9 |
| Adam | 5e-3 | 1e-4 | (0.9, 0.999) |
| AdamW | 1e-3 | 1e-3 | (0.9, 0.999) |

**HTJU settings:** `weight_decay_type='AdamW'`

---

## 5. Image Classification — ImageNet (ResNet-50)

**Model:** ResNet-50, batch_size=256, cosine LR, 150 epochs, 5 warmup epochs

### HTJU Paper Configs (Table 8)

| t_lr | h_lr | t_weight_decay | h_weight_decay | H_optim | momentum | t_betas |
|------|------|---------------|---------------|---------|----------|---------|
| 1e-3 | 1e-3 | 1e-3 | 1e-4 | SGD | 0.9 | (0.9, 0.999) |

### Baseline Configs

| Optimizer | lr | weight_decay |
|-----------|-----|-------------|
| SGD | 0.1 | 1e-4 |
| Adam | 1e-3 | 1e-4 |
| AdamW | 1e-3 | 1e-3 |

**HTJU settings:** `weight_decay_type='AdamW'`

---

## 6. Reinforcement Learning — PPO

**Environments:** Pendulum-v1, Ant-v5, HalfCheetah-v5, Walker2d-v5

### Optimizer Configs

| Optimizer | t_lr | h_lr | weight_decay | betas | H_optim | Notes |
|-----------|------|------|-------------|-------|---------|-------|
| HTJU | 3e-4 | 3e-4 | t:1e-3 / h:1e-3 | (0.9, 0.999) | Adam | weight_decay_type=AdamW, hessian_scale=0.05 |
| Adam | 3e-4 | — | 0 | (0.9, 0.999) | — | — |
| SGD | 3e-4 | — | 0 | — | — | momentum=0.9 |

### Environment-Specific Settings

| Environment | epochs | max_step | batch_steps | gamma | lambd |
|-------------|--------|----------|-------------|-------|-------|
| Pendulum-v1 | 500 | 200 | 2000 | 0.99 | 0.95 |
| Ant-v5 | 80000 | 1000 | 2000 | 0.99 | 0.95 |
| HalfCheetah-v5 | 75000 | 1000 | 2000 | 0.99 | 0.95 |
| Walker2d-v5 | 150000 | 1000 | 2000 | 0.99 | 0.95 |

**Common PPO settings:** epsilon=0.2, batch_size=128, hidden=[256, 256], 10 PPO epochs per update

---

## 7. Key Hyperparameter Guidelines

### Learning Rate
- **TJU:** Standard lr=1e-3 for most tasks
- **NdaTJU:** lr=1e-3 (small models), 2.5e-4 (large transformers), 5e-3 (U-KAN)
- **HTJU:** t_lr and h_lr typically equal; use CosineAnnealingLRDual

### Weight Decay
- **L2 type:** Traditional, adds to gradient (use with TJU)
- **AdamW type:** Decoupled, applies directly to parameters (use with NdaTJU, HTJU)
- **Stable type:** Scaled by denominator mean, less common
- Typical values: 1.2e-6 (LSTM), 0.02 (Transformer-XL), 5e-4 (Segmentation), 1e-3 (Classification)

### Hessian Scale
- Range: 0.0–1.0, default 0.05
- Lower values → less Hessian influence → closer to Adam
- Higher values → stronger curvature adaptation

### S-Weight Blending (HTJU)
- Sigmoid: `s = 1/(1+exp(-18*(epoch/total_epoch - 0.5)))`
- Early training: s≈0 → homotopy optimizer dominates (exploration)
- Late training: s≈1 → TJU branch dominates (exploitation)

### Warmup
- NdaTJU: 100–5000 steps (longer for large models)
- HTJU: 0 steps (s-weight provides natural exploration)

### Cosine Scheduler
- Use `CosineAnnealingLRDual` for HTJU (anneals both t_lr and h_lr)
- Post-schedule learning rate: base_lr * 0.01
