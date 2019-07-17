import torch


class IterLambdaScheduler(object):
    def __init__(self, optimizer, lr_func, last_iter=-1):
        if not isinstance(optimizer, torch.optim.Optimizer):
            return TypeError
        self.optimizer = optimizer
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for idx, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    return KeyError
        self.last_iter = last_iter
        self.lr_func = lr_func
        self.base_lrs = self.get_lr()

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def _get_new_lr(self):
        scale = self.lr_func(self.last_iter)
        return [scale * base_lr for base_lr in self.base_lrs]

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter
        for group, lr in zip(self.optimizer.param_groups, self._get_new_lr()):
            group['lr'] = lr
