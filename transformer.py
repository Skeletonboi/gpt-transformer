import torch
import torch.nn as nn

# The naming convention for the following GPT-2 architecture follows 
# that of the huggingface/original implementation so weights can be copied
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        pass
    def forward(self, x):
        pass

class MLP(nn.Module):
    def __init__(self, config):
        self.embed_size = config.get("embed_size")
        # n_neurons hyperparam set to 4 * embedding dim
        self.c_fc = nn.Linear(self.embed_size, 4*self.embed_size) 
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*self.embed_size, self.embed_size)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.get("embed_size")
            
        self.attn = self.MultiHeadAttention(config)
        self.ln_1 = nn.LayerNorm(self.embed_size)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(self.embed_size)

    def forward(self, x):
        # LayerNorm is applied BEFORE attn and mlp layers
        x = x + self.attn(self.ln_1(x))
        x=  x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.block_size = config.get("block_size")
        self.vocab_size = config.get("vocab_size")
        self.n_layers = config.get("n_layers")
        self.embed_size = config.get("embed_size")
        
        self.wte = nn.Embedding(self.vocab_size, self.embed_size) # token embedding
        self.wpe = nn.Embedding(self.block_size, self.embed_size) # positional embedding
        self.drop = nn.Dropout(0.1) # dropout - may not use unless overfitting

        self.h = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.n_layers)])
        self.ln_f = nn.LayerNorm(self.embed_size)
        self.out_proj = nn.Linear(self.embed_size, self.vocab_size, bias=False)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.wpe(pos) + self.wte(x) 
        # x = self.drop(x)
        for block in self.h:
            x = block(x) # transformer blocks
        x = self.ln_f(x) # layernorm at the end
        x =  self.out_proj(x) # project embd dim to vocab dim as logits
        return x

        


        