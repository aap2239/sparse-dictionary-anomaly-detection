import torch.nn as nn 
import torch 
from optim import * 


class DictLearn(nn.Module):
    def __init__(self, n_components, optim='ProxSGD'):
        super().__init__()
        self.n_components = n_components
        self.mse = nn.MSELoss()
        self.optim = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'ProxSGD': ProxSGD,
            'ProxAdam': ProxAdam
        }.get(optim, ProxSGD)
        self.prox = optim.startswith('Prox')

    def fit(self, X, a, epochs, lr=1e-3, eps=1e-8):
        X = X.reshape(len(X), -1)
        X = torch.as_tensor(X).float().cuda()

        self.mean = X.mean()
        self.std = X.std()
        self.eps = eps

        n, d = X.shape
        X = (X - self.mean) / (self.std + self.eps)

        linear = nn.Linear(self.n_components, n, bias=False).cuda()
        self.dict = nn.Embedding(self.n_components, d).cuda()
        #self.dict = nn.Linear(self.n_components, d, bias=False).cuda()

        params = list(linear.parameters()) + list(self.dict.parameters())
        args = {'lr': lr}
        if self.prox: args['a'] = a
        optim = self.optim(params, **args)
        hists = {'mse': [], 'l1': [], 'loss': []}

        for e in range(epochs):
            p = linear(self.dict.weight.t()).t()
            optim.zero_grad()
            mse = self.mse(p, X)
            l1 = a * torch.norm(linear.weight, p=1, dim=-1).sum()
            l = mse
            if not self.prox: l = l + l1
            l = mse + l1
            l.backward()
            optim.step()
            if self.prox: l = l + l1

            hists['mse'].append(mse.item())
            hists['l1'].append(l1.item())
            hists['loss'].append(l.item())

        self.hists = hists

        return self

    def transform(self, y, a, epochs, lr=1e-3):
        y = torch.as_tensor(y.reshape(len(y), -1)).float().cuda()
        D = self.dict.weight.detach()

        n, d = D.shape
        m, _ = y.shape

        linear = nn.Linear(n, m, bias=False).cuda()
        # D = (D - self.mean) / (self.std + self.eps)
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
            l = mse + l1
            l.backward()
            optim.step()
            if self.prox: l = l + l1

            hists['mse'].append(mse.item())
            hists['l1'].append(l1.item())
            hists['loss'].append(l.item())

        return linear.weight.detach().cpu(), hists
