from optim import *
import torch
import torch.nn as nn


class LASSO(nn.Module):
    def __init__(self, optim='ProxSGD'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.optim = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'ProxGD': ProxGD,
            'ProxSGD': ProxSGD,
            'ProxAdam': ProxAdam
        }[optim]
        self.prox = optim.startswith('Prox')

    def fit(self, D, eps=1e-8):
        D = D.reshape(len(D), -1)
        self.D = torch.as_tensor(D).float().cuda()
        self.mean = 0 #self.D.mean()
        self.std = self.D.std()
        self.eps = eps
        return self

    def transform(self, y, a, epochs, lr=1e-3):
        y = torch.as_tensor(y.reshape(len(y), -1)).float().cuda()
        D = self.D

        n, d = self.D.shape
        m, _ = y.shape

        linear = nn.Linear(n, m, bias=False).cuda()
        D = (D - self.mean) / (self.std + self.eps)
        y = (y - self.mean) / (self.std + self.eps)

        args = {'lr': lr}
        if self.prox: args['a'] = a
        optim = self.optim(linear.parameters(), **args)
        hists = {'mse': [], 'l1': [], 'loss': []}

        for e in range(epochs):
            p = linear(D.t()).t()
            optim.zero_grad()
            mse = self.mse(p, y)
            l1 = a * torch.norm(linear.weight, p=1, dim=-1).sum()
            l = mse
            if not self.prox: l = l + l1
            l.backward()
            optim.step()
            if self.prox: l = l + l1

            hists['mse'].append(mse.item())
            hists['l1'].append(l1.item())
            hists['loss'].append(l.item())

        return linear.weight.detach().cpu(), hists
