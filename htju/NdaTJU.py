import torch
import math
from torch.optim.optimizer import Optimizer


class NdaTJU(Optimizer):
    r"""NdaTJU (NdaTJU) — TJUv3 with AdamW-style decoupled weight decay.

    Key improvements over TJUv3:
    1. AdamW-type decoupled weight decay: weight decay is applied directly
       to parameters, scaled by the current learning rate
    2. Retains TJUv3's approximate Hessian logic and numerical stability
    3. Maintains original parameter update ordering

    Also referred to as NdaTJU in the HTJU paper.

    Args:
        params (iterable): Model parameters to optimize.
        lr (float): Base learning rate. (default: 1e-3)
        betas (tuple(float, float)): Coefficients for gradient and squared
            gradient EMA. (default: (0.9, 0.999))
        beta_h (float): Decay factor for approximate Hessian. (default: 0.85)
        eps (float): Numerical stability constant. (default: 1e-8)
        rebound (str): Hessian bound mode: ``'constant'`` | ``'belief'``.
            (default: 'constant')
        warmup (int): Number of warmup steps. (default: 100)
        init_lr (float): Initial LR for warmup. (default: lr/1000)
        weight_decay (float): Weight decay coefficient. (default: 0.0)
        weight_decay_type (str): ``'L2'`` | ``'stable'`` | ``'AdamW'``.
            (default: 'AdamW')
        hessian_scale (float): Scaling factor for approximate Hessian.
            (default: 0.05)
        total_steps (int): Total training steps for cosine annealing. (default: 10000)
        use_cosine_scheduler (bool): Whether to use cosine annealing. (default: True)
    """

    def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            beta_h=0.85,
            eps=1e-8,
            rebound='constant',
            warmup=100,
            init_lr=None,
            weight_decay=0.0,
            weight_decay_type='AdamW',
            hessian_scale=0.05,
            total_steps=10000,
            use_cosine_scheduler=True
    ):
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if weight_decay_type not in ['L2', 'stable', 'AdamW']:
            raise ValueError(f"Invalid weight_decay_type: {weight_decay_type}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta_h=beta_h,
            eps=eps,
            rebound=rebound,
            warmup=warmup,
            init_lr=init_lr or lr / 1000.0,
            base_lr=lr,
            weight_decay=weight_decay,
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
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("NdaTJU does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['approx_hessian'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']

                # Learning rate schedule
                current_lr = self._compute_lr(group, step)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                approx_hessian = state['approx_hessian']

                # (1) L2 weight decay
                if group['weight_decay_type'] == 'L2' and group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # (2) Update momentum terms
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # (3) Bias correction
                bias_corr1 = 1 - beta1 ** step
                bias_corr2 = 1 - beta2 ** step
                step_size = current_lr / bias_corr1

                # (4) Approximate Hessian
                delta_grad = grad - (exp_avg / bias_corr1)
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
                ).add_(group['eps'])

                # (6) Compute update direction
                update = exp_avg / denom

                # (7) Stable weight decay
                if group['weight_decay_type'] == 'stable' and group['weight_decay'] != 0:
                    decay_factor = group['weight_decay'] / denom.mean().clamp(min=1e-8)
                    update.add_(p, alpha=decay_factor)

                # (8) AdamW decoupled weight decay
                if group['weight_decay_type'] == 'AdamW' and group['weight_decay'] != 0:
                    p.data.mul_(1 - group['weight_decay'] * current_lr)

                # (9) Apply parameter update
                p.add_(update, alpha=-step_size)

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
