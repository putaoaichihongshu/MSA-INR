import torch
import torch.nn as nn

class SineActivation(nn.Module):
    def __init__(self, s_min, s_max):
        super().__init__()
        self.s_min = s_min
        self.s_max = s_max
    def forward(self, x):
        size = x.size(1)
        device = x.device
        freqs = torch.logspace(self.s_min, self.s_max, steps=size, device=device)
        amps = 1.0 / freqs
        return amps * torch.sin(freqs * x)


class MSANet(nn.Module):
    def __init__(self,
                 in_features=3,
                 hidden_features=256,
                 hidden_layers=2,
                 out_features=1,
                 s_min=None,   # 新增：每层一个s_min
                 s_max=None    # 新增：每层一个s_max
                 ):
        super().__init__()
        # 默认每层取1和5.0
        if s_min is None:
            s_min = [1.0] * hidden_layers
        if s_max is None:
            s_max = [5.0] * hidden_layers
        assert len(s_min) == hidden_layers, "s_min必须和隐藏层数对应"
        assert len(s_max) == hidden_layers, "s_max必须和隐藏层数对应"

        layers = []
        # 第一层
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(SineActivation(s_min[0], s_max[0]))
        # 隐藏层-2
        for i in range(1, hidden_layers-1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(SineActivation(s_min[i], s_max[i]))
        # 倒数第二层
        layers.append(nn.Linear(hidden_features, hidden_features))
        layers.append(SineActivation(s_min[hidden_layers-1], s_max[hidden_layers-1]))
        # 输出层
        layers.append(nn.Linear(hidden_features, out_features))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_model(**kwargs):
    return MSANet(**kwargs)
