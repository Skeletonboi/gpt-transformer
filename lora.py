import torch
import torch.nn as nn
import bitsandbytes as bnb

class LowRankLayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        super().__init__()
        self.scale = alpha/rank
        
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x)) * self.scale

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.lora = LowRankLayer(in_features, out_features, rank, alpha)

        self.linear.weight.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)

class QLoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha, bias=True):
        super().__init__()
        self.linear = bnb.nn.LinearNF4(in_features, out_features, bias=bias)
        self.lora = LowRankLayer(in_features, out_features, rank, alpha)

        self.linear.weight.requires_grad = False

    def forward(self, x):
        return self.linear(x) + self.lora(x)