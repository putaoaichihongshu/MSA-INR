import torch
import torch.nn as nn
import math

class FourierFeatureMapping(nn.Module):
    def __init__(self, in_features, mapping_size=256, scale=10.0):
        super().__init__()
        # 随机采样一个变换矩阵B：mapping_size x in_features
        self.B = nn.Parameter(
            torch.randn((in_features, mapping_size)) * scale,
            requires_grad=False
        )

    def forward(self, x):
        # x: [N, in_features]
        # f(x) = [sin(Bx), cos(Bx)]
        x_proj = 2 * math.pi * x @ self.B  # [N, mapping_size]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class FourierMLP(nn.Module):
    def __init__(self, in_features=3, out_features=1, mapping_size=256, hidden_features=256, hidden_layers=2, scale=10.0):
        super().__init__()
        self.fourier = FourierFeatureMapping(in_features, mapping_size, scale)
        input_dim = mapping_size * 2  # sin+cos
        layers = [nn.Linear(input_dim, hidden_features), nn.ReLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features, out_features))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.fourier(x)
        return self.mlp(x)

def get_model(in_features=3, hidden_features=256, hidden_layers=2, out_features=1,
              mapping_size=256, scale=10.0, **kwargs):
    """
    兼容你的主文件的get_model接口
    """
    return FourierMLP(
        in_features=in_features,
        out_features=out_features,
        mapping_size=mapping_size,
        hidden_features=hidden_features,
        hidden_layers=hidden_layers,
        scale=scale,
    )
