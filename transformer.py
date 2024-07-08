import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# The naming convention for the following GPT-2 architecture follows 
# that of the huggingface/original implementation so weights can be copied
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.get("n_embed")
        self.n_context = config.get("n_context")
        self.n_head = config.get("n_head")
        self.head_size = self.n_embed//self.n_head
        # huggingface combines QKV projection matrices into single matrix
        self.c_attn = nn.Linear(self.n_embed, 3*self.n_embed) 
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        # store attn mask as buffer to keep it on GPU 
        self.register_buffer("mask", 
                             torch.tril(torch.ones(self.head_size, self.head_size))\
                                .expand(size=(1,1,self.head_size, self.head_size))
                             )
        
    def forward(self, x):
        B, S, E = x.size() # batch_size, seq_length, embed_size
        assert E == self.n_embed
        assert S <= self.n_context

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2) # split into BxSxE outputs
        # split embed dim into (n_head x embd//n_head) for each "head, (B,n_head) are now batch dims
        q = q.view(B, S, self.n_head, self.head_size).transpose(1,2) # (B,n_head,S,head_size)
        k = k.view(B, S, self.n_head, self.head_size).transpose(1,2) # (B,n_head,S,head_size)
        v = v.view(B, S, self.n_head, self.head_size).transpose(1,2) # (B,n_head,S,head_size)

        attn = (q @ k.transpose(2,3))/(math.sqrt(self.head_size)) # (B,n_head,S,S) matrix
        attn = attn.masked_fill(self.mask[:,:,:S,:S] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = attn @ v # (B,n_head,S,head_size)
        out = out.transpose(1,2).view(B,S,E) # (B,S,n_head,head_size) -> (B,S,E)
        proj_out = self.c_proj(out)

        return proj_out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.get("n_embed")
        # n_neurons hyperparam set to 4 * embedding dim
        self.c_fc = nn.Linear(self.n_embed, 4*self.n_embed) 
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4*self.n_embed, self.n_embed)

    def forward(self, x):
        x = self.gelu(self.c_fc(x))
        x = self.c_proj(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.get("n_embed")
            
        self.attn = self.MultiHeadAttention(config)
        self.ln_1 = nn.LayerNorm(self.n_embed)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(self.n_embed)

    def forward(self, x):
        # LayerNorm is applied BEFORE attn and mlp layers
        x = x + self.attn(self.ln_1(x))
        x=  x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_context = config.get("n_context")
        self.n_vocab = config.get("n_vocab")
        self.n_layers = config.get("n_layers")
        self.n_embed = config.get("n_embed")
        
        self.wte = nn.Embedding(self.n_vocab, self.n_vocab) # token embedding
        self.wpe = nn.Embedding(self.n_context, self.n_context) # positional embedding
        self.drop = nn.Dropout(0.1) # dropout - may not use unless overfitting

        self.h = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.n_layers)])
        self.ln_f = nn.LayerNorm(self.n_embed)
        self.out_proj = nn.Linear(self.n_embed, self.n_vocab, bias=False)
    
    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.wpe(pos) + self.wte(x) 
        # x = self.drop(x)
        for block in self.h:
            x = block(x) # transformer blocks
        x = self.ln_f(x) # layernorm at the end
        x =  self.out_proj(x) # project embd dim to vocab dim as logits
        return x

        


        