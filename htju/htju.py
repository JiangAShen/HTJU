import torch
import math
from torch.optim.optimizer import Optimizer


class HTJU(Optimizer):
    r"""HTJU — Hybrid TJU: the final optimizer in the TJU family.

    HTJU combines the dual-optimizer framework of ATJU (TJU + homotopy optimizer
    with sigmoid blending weight ``s``) with the improvements from NdaTJU:
    - AdamW-style decoupled weight decay
    - Cosine annealing learning rate scheduler
    - Refined approximate Hessian with additive fusion (hessian_scale=0.05)
    - Sigmoid blending schedule: s = 1/(1+exp(-18*(epoch/total_epoch - 0.5)))

    Args:
        params (iterable): Model parameters to optimize.
        t_lr (float): Base learning rate for the TJU branch. (default: 1e-3)
        h_lr (float): Base learning rate for homotopy optimizer. (default: 1e-3)
        t_betas (tuple): Adam-style (beta1, beta2) for TJU branch. (default: (0.9, 0.999))
        h_betas (tuple): (beta1, beta2) for homotopy Adam/AdamW. (default: (0.9, 0.999))
        beta_h (float): Decay factor for approximate Hessian. (default: 0.85)
        t_eps (float): Epsilon for TJU. (default: 0)
        h_eps (float): Epsilon for homotopy optimizer. (default: 0)
        momentum (float): Momentum for homotopy SGD. (default: 0)
        rebound (str): Hessian bound mode: ``'constant'`` | ``'belief'``.
            (default: 'constant')
        warmup (int): Number of warmup steps. (default: 0)
        init_lr (float): Initial LR for warmup. (default: t_lr/1000)
        t_weight_decay (float): Weight decay for TJU branch. (default: 0.0)
        h_weight_decay (float): Weight decay for homotopy branch. (default: 0)
        weight_decay_type (str): ``'L2'`` | ``'stable'`` | ``'AdamW'``.
            (default: 'AdamW')
        H_optim (str or None): Homotopy optimizer: ``'SGD'`` | ``'Adam'`` | ``'AdamW'``.
            (default: None)
        total_epoch (int): Total training epochs for s-weight schedule.
        epoch_now (int): Current epoch number.
        hessian_scale (float): Scaling factor for approximate Hessian. (default: 0.05)
        total_steps (int): Total steps for cosine annealing. (default: 100000)
        use_cosine_scheduler (bool): Whether to use cosine annealing. (default: False)
    """

    def __init__(
            self,
            params,
            t_lr=1e-3,
            h_lr=1e-3,
            t_betas=(0.9, 0.999),
            h_betas=(0.9, 0.999),
            beta_h=0.85,
            t_eps=1e-8,
            h_eps=1e-8,
            momentum=0.9,
            rebound='constant',
            warmup=0,
            init_lr=None,
            t_weight_decay=5e-4,
            h_weight_decay=5e-4,
            weight_decay_type='AdamW',
            H_optim=None,
            total_epoch=0,
            epoch_now=0,
            hessian_scale=0.05,
            total_steps=100000,
            use_cosine_scheduler=False
    ):
        self.epoch_now = epoch_now
        self.total_epoch = total_epoch
        self.step_now = 0

        if not 0.0 <= t_weight_decay:
            raise ValueError(f"Invalid t_weight_decay: {t_weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")
        if H_optim not in ['SGD', 'Adam', 'AdamW', None]:
            raise ValueError(f"Invalid H_optim: {H_optim}")
        if not 0.0 <= total_epoch:
            raise ValueError(f"Invalid total_epoch: {total_epoch}")
        if not 0.0 <= epoch_now:
            raise ValueError(f"Invalid epoch_now: {epoch_now}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum: {momentum}")

        defaults = dict(
            t_lr=t_lr,
            h_lr=h_lr,
            t_betas=t_betas,
            h_betas=h_betas,
            beta_h=beta_h,
            t_eps=t_eps,
            h_eps=h_eps,
            momentum=momentum,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr or t_lr / 1000.0,
            base_lr=t_lr,
            t_weight_decay=t_weight_decay,
            h_weight_decay=h_weight_decay,
            H_optim=H_optim,
            weight_decay_type=weight_decay_type,
            hessian_scale=hessian_scale,
            total_steps=total_steps,
            use_cosine_scheduler=use_cosine_scheduler
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            t_beta1, t_beta2 = group['t_betas']
            h_beta1, h_beta2 = group["h_betas"]
            h_lr = group["h_lr"]
            t_lr = group["t_lr"]
            h_eps = group["h_eps"]
            momentum = group["momentum"]
            h_wd_coef = group["h_weight_decay"]
            H_optim = group['H_optim']

            # Compute sigmoid blending weight s
            if self.epoch_now == 1:
                self.total_epoch = self.total_epoch * (self.step_now + 1)
                self.epoch_now += 1

            if self.epoch_now >= 1:
                s = 1 / (1 + math.exp(-18 * (self.step_now / self.total_epoch - 0.15)))
            else:
                s = 0

            """
            rl
            if self.epoch_now >= 1:
                s = 1 / (1 + math.exp(-18 * (self.epoch_now / self.total_epoch - 0.5)))
            else:
                s = 0
            """
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("HTJU does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_grad_H'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq_grad_H"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                state['step'] += 1
                step = state['step']
                self.step_now = step

                current_lr = t_lr

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']
                momentum_buffer = state["momentum_buffer"]
                exp_avg_grad_H = state['exp_avg_grad_H']
                exp_avg_sq_grad_H = state["exp_avg_sq_grad_H"]

                grad_TJU = p.grad
                grad_H = p.grad

                # (1) L2 weight decay
                if group['weight_decay_type'] == 'L2' and group['t_weight_decay'] != 0:
                    grad_TJU = grad_TJU.add(p, alpha=group['t_weight_decay'])

                # (2) Update TJU momentum terms
                exp_avg.mul_(t_beta1).add_(grad_TJU, alpha=1 - t_beta1)
                exp_avg_sq.mul_(t_beta2).addcmul_(grad_TJU, grad_TJU, value=1 - t_beta2)

                # (3) Bias correction
                bias_corr1 = 1 - t_beta1 ** step
                bias_corr2 = 1 - t_beta2 ** step
                step_size = current_lr / bias_corr1

                # (4) Approximate Hessian
                delta_grad = grad_TJU - (exp_avg / bias_corr1)
                approx_hessian.mul_(group['beta_h']).addcmul_(
                    delta_grad, delta_grad, value=1 - group['beta_h'])

                if group['rebound'] == 'constant':
                    denom_hessian = approx_hessian.abs().clamp_(min=1e-3)
                else:
                    bound_val = max(delta_grad.norm(p=float('inf')).item(), 1e-5)
                    denom_hessian = torch.max(approx_hessian.abs(),
                                              torch.tensor(bound_val, device=p.device))

                # (5) Combined denominator
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(
                    group['hessian_scale'] * denom_hessian,
                    alpha=1.0
                ).add_(group['t_eps'])

                # (6) TJU update direction
                new_update = exp_avg / denom

                # (7) Stable weight decay
                if group['weight_decay_type'] == 'stable' and group['t_weight_decay'] != 0:
                    decay_factor = group['t_weight_decay'] / denom.mean().clamp(min=1e-8)
                    new_update.add_(p, alpha=decay_factor)

                # (8) AdamW decoupled weight decay — scaled by (1-s) so TJU
                #     weight decay fades as the homotopy branch takes over.
                if group['weight_decay_type'] == 'AdamW' and group['t_weight_decay'] != 0:
                    p.data.mul_(1 - (1 - s) * group['t_weight_decay'] * current_lr)

                update_TJU = (1 - s) * new_update

                # === Homotopy Optimizer Branch ===
                if H_optim == 'SGD':
                    grad_H = grad_H.add(p, alpha=h_wd_coef)
                    update_H = s * (grad_H + momentum * momentum_buffer)
                    state["momentum_buffer"] = grad_H + momentum * momentum_buffer

                elif H_optim == 'Adam':
                    grad_H = grad_H.add(p, alpha=h_wd_coef)
                    exp_avg_grad_H = h_beta1 * exp_avg_grad_H + (1 - h_beta1) * grad_H
                    exp_avg_sq_grad_H = h_beta2 * exp_avg_sq_grad_H + (1 - h_beta2) * grad_H ** 2

                    exp_avg_grad_H_hat = exp_avg_grad_H / (1 - h_beta1 ** step)
                    exp_avg_sq_grad_H_hat = exp_avg_sq_grad_H / (1 - h_beta2 ** step)
                    update_H = s * (exp_avg_grad_H_hat / (torch.sqrt(exp_avg_sq_grad_H_hat) + h_eps))

                    state["exp_avg_grad_H"] = exp_avg_grad_H
                    state["exp_avg_sq_grad_H"] = exp_avg_sq_grad_H

                elif H_optim == 'AdamW':
                    exp_avg_grad_H = h_beta1 * exp_avg_grad_H + (1 - h_beta1) * grad_H
                    exp_avg_sq_grad_H = h_beta2 * exp_avg_sq_grad_H + (1 - h_beta2) * grad_H ** 2

                    exp_avg_grad_H_hat = exp_avg_grad_H / (1 - h_beta1 ** step)
                    exp_avg_sq_grad_H_hat = exp_avg_sq_grad_H / (1 - h_beta2 ** step)
                    update_H = s * ((exp_avg_grad_H_hat / (torch.sqrt(exp_avg_sq_grad_H_hat) + h_eps)) + h_wd_coef * p)

                    state["exp_avg_grad_H"] = exp_avg_grad_H
                    state["exp_avg_sq_grad_H"] = exp_avg_sq_grad_H
                else:
                    update_H = s * 0

                # === Combined Update ===
                update = -step_size * update_TJU - h_lr * update_H
                p.add_(update)

        return loss

    def _compute_lr(self, group, step):
        """Cosine annealing learning rate schedule with warmup."""
        if step <= group['warmup']:
            return group['init_lr'] + (group['base_lr'] - group['init_lr']) * step / group['warmup']

        if not group['use_cosine_scheduler']:
            return group['base_lr']

        t = step - group['warmup']
        T = group['total_steps'] - group['warmup']
        if t <= T:
            return group['base_lr'] * (0.5 * (1 + math.cos(math.pi * t / T)))
        return group['base_lr'] * 0.01
