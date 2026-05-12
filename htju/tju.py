import torch
from torch.optim.optimizer import Optimizer


class TJU(Optimizer):
    r"""TJU Optimizer — Adaptive gradient method with approximate diagonal Hessian.

    TJU maintains an exponential moving average of gradients and an approximate
    diagonal Hessian via the outer product of parameter updates, providing
    curvature-aware preconditioning without second-order derivative computation.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        beta (float, optional): coefficient for computing running averages of gradient (default: 0.9)
        eps (float, optional): term added to the denominator to improve numerical stability (default: 1e-4)
        rebound (str, optional): mode for bounding the diagonal Hessian:
            ``'constant'`` | ``'belief'`` (default: 'constant')
        warmup (int, optional): number of warmup steps (default: 500)
        init_lr (float, optional): initial learning rate for warmup (default: lr/1000)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        weight_decay_type (str, optional): type of weight decay:
            ``'L2'`` | ``'decoupled'`` | ``'stable'`` (default: 'L2')
    """

    def __init__(self, params, lr=1e-3, beta=0.9, eps=1e-4, rebound='constant',
                 warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None):
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate value: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if rebound not in ['constant', 'belief']:
            raise ValueError(f"Invalid rebound mode: {rebound}")
        if not 0.0 <= warmup:
            raise ValueError(f"Invalid warmup steps: {warmup}")
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError(f"Invalid initial learning rate: {init_lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if weight_decay_type is None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError(f"Invalid weight decay type: {weight_decay_type}")

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['approx_hessian'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['update'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Calculate current lr (linear warmup)
                if state['step'] < group['warmup']:
                    curr_lr = (group['base_lr'] - group['init_lr']) * state['step'] / group['warmup'] + group['init_lr']
                else:
                    curr_lr = group['lr']

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('TJU does not support sparse gradients.')

                # L2 weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] == 'L2':
                    grad = grad.add(p, alpha=group['weight_decay'])

                beta = group['beta']
                eps = group['eps']
                exp_avg_grad = state['exp_avg_grad']
                B = state['approx_hessian']
                d_p = state['update']

                state['step'] += 1
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction

                # Compute gradient difference
                delta_grad = grad - exp_avg_grad
                if group['rebound'] == 'belief':
                    rebound = delta_grad.norm(p=float('inf')).item()
                else:
                    rebound = 0.01
                    eps = eps / rebound

                # Update running average of gradient
                exp_avg_grad.add_(delta_grad, alpha=alpha)

                # Update approximate Hessian via outer product of update directions
                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - B.mul(v_sq).sum()

                B.addcmul_(v_sq, delta)

                # Compute parameter update direction
                if group['rebound'] == 'belief':
                    denom = torch.max(B.abs(), torch.tensor(rebound, device=p.device)).add_(eps / alpha)
                else:
                    denom = B.abs().clamp_(min=rebound)

                d_p.copy_(exp_avg_grad.div(denom))

                # Apply non-L2 weight decay
                if group['weight_decay'] != 0 and group['weight_decay_type'] != 'L2':
                    if group['weight_decay_type'] == 'stable':
                        weight_decay = group['weight_decay'] / denom.mean().item()
                    else:
                        weight_decay = group['weight_decay']
                    d_p.add_(p, alpha=weight_decay)

                p.add_(d_p, alpha=-curr_lr)

        return loss
