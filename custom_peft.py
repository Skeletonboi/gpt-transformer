import torch
import torch.nn as nn
import torch.nn.functional as F
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
    
class DoRALayer(nn.Module):
    """
    DoRA is an evolution of LoRA which decomposes the original pre-trained weight matrix
    into a magnitude vector and direction matrix, upon which LoRA is applied to the direction matrix.
    The low-rank matrices and the magnitude vector are then jointly trained/fintuned.
    """
    def __init__(self, module, rank, alpha, bias=True):
        super().__init__()
        self.linear = nn.Linear(module.in_features, module.out_features, bias=bias)
        self.lora = LowRankLayer(module.in_features, module.out_features, rank, alpha)
        self.magnitude = nn.Parameter(module.weight.norm(p=2, dim=1, keepdim=True), requires_grad=True)

        self.linear.weight.requires_grad = False

    def forward(self, x):
        merged_weights = self.linear.weight + (self.lora.B.weight @ self.lora.A.weight) * self.lora.scale
        merged_weights = merged_weights / merged_weights.norm(p=2, dim=1, keepdim=True)
        merged_weights = self.magnitude * merged_weights
        return  F.linear(x, merged_weights, bias=self.linear.bias)

class QDoRALayer(nn.Module):
    """
    DoRA with pre-trained linear weights quantized to NF4,
    manually dequantized for weight-merging.
    """
    def __init__(self, module, rank, alpha, bias=True):
        super().__init__()
        self.linear = bnb.nn.LinearNF4(module.in_features, module.out_features, bias=bias)
        self.lora = LowRankLayer(module.in_features, module.out_features, rank, alpha)
        self.magnitude = nn.Parameter(module.weight.norm(p=2, dim=1, keepdim=True), requires_grad=True)

        self.linear.weight.requires_grad = False

    def forward(self, x):
        merged_weights = bnb.functional.dequantize_nf4(self.linear.weight, self.linear.weight.quant_state)
        merged_weights = merged_weights + (self.lora.B.weight @ self.lora.A.weight) * self.lora.scale
        merged_weights = merged_weights / merged_weights.norm(p=2, dim=1, keepdim=True)
        merged_weights = merged_weights * self.magnitude
        return  F.linear(x, merged_weights, bias=self.linear.bias)