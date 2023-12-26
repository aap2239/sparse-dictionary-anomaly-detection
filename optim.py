# ref: https://github.com/rahulkidambi/AccSGD.git
import torch


class ProxGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, a=0.0, weight_decay=0.0):
        defaults = {'lr': lr, 'a': a, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super(AccSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            a = group['a']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                param_state = self.state[p]
                grad = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                # sgd
                p.data.add_(-lr, p.grad.data)
                # prox
                mask = torch.zeros_like(p.data, device=p.data.device)
                mask[p.data > a] = -1
                mask[p.data < -a] = 1
                p.data.add_(a, mask)

        return loss

class ProxSGD(torch.optim.SGD):
    def __init__(self, params, lr=1e-3, a=0.0, weight_decay=0.0):
        defaults = {'lr': lr, 'weight_decay': weight_decay}
        super().__init__(params, **defaults)
        self.a = a

    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            a = self.a
            for p in group['params']:
                # prox
                mask = torch.zeros_like(p.data, device=p.data.device)
                mask[p.data > a] = -1
                mask[p.data < -a] = 1
                p.data.add_(a, mask)

        return loss

class ProxAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, a=0.0, weight_decay=0.0):
        defaults = {'lr': lr, 'weight_decay': weight_decay}
        super().__init__(params, **defaults)
        self.a = a

    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            a = self.a
            for p in group['params']:
                # prox
                mask = torch.zeros_like(p.data, device=p.data.device)
                mask[p.data > a] = -1
                mask[p.data < -a] = 1
                p.data.add_(a, mask)

        return loss
