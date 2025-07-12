import torch
import torch.nn as nn

class RowdyActivation(nn.Module):
    def __init__(self, act='tanh', K=5, n=10):
        super().__init__()
        self.K = K
        self.n = n
        if act == 'tanh':
            self.act1 = torch.tanh
        elif act == 'relu':
            self.act1 = torch.relu
        elif act == 'cos':
            self.act1 = torch.cos
        elif act == 'sigmoid':
            self.act1 = torch.sigmoid
        else:
            raise NotImplementedError(f"Unknown act: {act}")
        self.register_buffer('a1', torch.ones(1))
        self.register_buffer('x1', torch.ones(1))
        self.a = nn.Parameter(torch.zeros(K-1))
        self.x = nn.Parameter(torch.ones(K-1))
    def forward(self, x):
        out = self.a1 * self.act1(self.x1 * x)
        for k in range(2, self.K+1):
            ak = self.a[k-2]
            xk = self.x[k-2]
            out = out + ak * torch.sin((k-1) * self.n * xk * x)
        return out

class RowdyNet(nn.Module):
    def __init__(self,
                 in_features=3,
                 out_features=1,
                 hidden_features=256,
                 hidden_layers=3,
                 act='tanh',
                 K=5,
                 n=10):
        super().__init__()
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.layers.append(nn.Linear(in_features, hidden_features))
        self.acts.append(RowdyActivation(act, K, n))
        for _ in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_features, hidden_features))
            self.acts.append(RowdyActivation(act, K, n))
        self.out_layer = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        for l, act in zip(self.layers, self.acts):
            x = act(l(x))
        return self.out_layer(x)

def get_model(**kwargs):
    return RowdyNet(**kwargs)
