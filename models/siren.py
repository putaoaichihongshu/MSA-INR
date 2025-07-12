import torch
import torch.nn as nn
import math

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        in_features = self.linear.in_features
        if self.is_first:
            nn.init.uniform_(self.linear.weight, -1.0 / in_features, 1.0 / in_features)
        else:
            bound = math.sqrt(6.0 / in_features) / self.omega_0
            nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIRENNet(nn.Module):
    def __init__(self, in_features=3, hidden_features=256, hidden_layers=2, out_features=1,
                 first_omega_0=30, hidden_omega_0=30, outermost_linear=True):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(SineLayer(in_features, hidden_features, omega_0=first_omega_0, is_first=True))
        for _ in range(hidden_layers - 1):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega_0))
        self.final_linear = nn.Linear(hidden_features, out_features)
        nn.init.uniform_(self.final_linear.weight,
                         -math.sqrt(6.0 / hidden_features) / hidden_omega_0,
                         math.sqrt(6.0 / hidden_features) / hidden_omega_0)
        nn.init.zeros_(self.final_linear.bias)
        self.outermost_linear = outermost_linear

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        x = self.final_linear(x)
        return x

def get_model(**kwargs):
    return SIRENNet(**kwargs)
