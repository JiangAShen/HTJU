import math


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler for single-LR optimizers (NdaTJU).

    Args:
        optimizer: Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. (default: 0)
        last_epoch (int): The index of the last epoch. (default: -1)
    """

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1

        if self.T_max == 0:
            step_ratio = 1.0
        else:
            step_ratio = self.last_epoch / self.T_max

        cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            new_lr = self.eta_min + (base_lr - self.eta_min) * cosine_factor
            param_group['lr'] = new_lr

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class CosineAnnealingLRDual:
    """Cosine annealing scheduler for dual-LR optimizers (ATJU, HTJU).

    Simultaneously anneals both the TJU learning rate (``t_lr``) and the
    homotopy optimizer learning rate (``h_lr``) following a cosine curve.

    Args:
        optimizer: Wrapped ATJU or HTJU optimizer.
        T_max (int): Maximum number of iterations (typically number of epochs).
        t_lr_min (float): Minimum learning rate for the TJU branch. (default: 0)
        h_lr_min (float): Minimum learning rate for the homotopy branch. (default: 0)
        last_epoch (int): The index of the last epoch. (default: -1)
    """

    def __init__(self, optimizer, T_max, t_lr_min=0, h_lr_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self.t_lr_min = t_lr_min
        self.h_lr_min = h_lr_min
        self.last_epoch = last_epoch
        self.t_lrs = [group['t_lr'] for group in optimizer.param_groups]
        self.h_lrs = [group['h_lr'] for group in optimizer.param_groups]

    def step(self):
        self.last_epoch += 1

        if self.T_max == 0:
            step_ratio = 1.0
        else:
            step_ratio = self.last_epoch / self.T_max

        cosine_factor = 0.5 * (1 + math.cos(math.pi * step_ratio))

        for param_group, t_lr_init in zip(self.optimizer.param_groups, self.t_lrs):
            new_lr = self.t_lr_min + (t_lr_init - self.t_lr_min) * cosine_factor
            param_group['t_lr'] = new_lr

        for param_group, h_lr_init in zip(self.optimizer.param_groups, self.h_lrs):
            new_h_lr = self.h_lr_min + (h_lr_init - self.h_lr_min) * cosine_factor
            param_group['h_lr'] = new_h_lr

    def get_t_lr(self):
        return [group['t_lr'] for group in self.optimizer.param_groups]

    def get_h_lr(self):
        return [group['h_lr'] for group in self.optimizer.param_groups]

    # Backward-compatible aliases
    def get_tju_lr(self):
        return self.get_t_lr()

    def get_A_optim_lr(self):
        return self.get_h_lr()
