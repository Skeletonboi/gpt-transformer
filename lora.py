import torch
import torch.nn as nn

class LowRankLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.scale = self.alpha/self.rank
        
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.lora = LowRankLayer(in_features, out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)