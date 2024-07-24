import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
import yaml

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension

# The naming convention for the following GPT-2 architecture follows 
# that of the huggingface/original implementation so weights can be copied
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_embed = config.get("n_embed")
        self.n_context = config.get("n_context")
        self.n_head = config.get("n_head")
        self.use_flash_attn = config.get("use_flash_attn")

        self.head_size = self.n_embed//self.n_head
        # huggingface combines QKV projection matrices into single matrix
        self.c_attn = nn.Linear(self.n_embed, 3*self.n_embed) 
        self.c_proj = nn.Linear(self.n_embed, self.n_embed)
        # store attn mask as buffer to keep it on GPU 
        self.register_buffer("mask", 
                             torch.tril(torch.ones(self.n_context, self.n_context))\
                                .expand(size=(1,1,self.n_context, self.n_context))
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

        if not self.use_flash_attn:
            attn = (q @ k.transpose(2,3))/(math.sqrt(self.head_size)) # (B,n_head,S,S) matrix
            attn = attn.masked_fill(self.mask[:,:,:S,:S] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            out = attn @ v # (B,n_head,S,head_size)
        else:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True) # pytorch built-in flash attention
        out = out.transpose(1,2).reshape(B,S,E) # (B,S,n_head,head_size) -> (B,S,E)
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
        self.use_dropout = config.get("use_dropout")
            
        self.attn = MultiHeadAttention(config)
        self.ln_1 = nn.LayerNorm(self.n_embed)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(self.n_embed)
        self.drop_attn = nn.Dropout(0.1)
        self.drop_mlp = nn.Dropout(0.1)

    def forward(self, x):
        # LayerNorm is applied BEFORE attn and mlp layers
        if self.use_dropout:
            x = x + self.drop_attn(self.attn(self.ln_1(x)))
            x =  x + self.drop_mlp(self.mlp(self.ln_2(x)))
        else:
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.config = config
        self.device = device
        self.n_context = config.get("n_context")
        self.n_vocab = config.get("n_vocab")
        self.n_layer = config.get("n_layer")
        self.n_embed = config.get("n_embed")
        self.use_dropout = config.get("use_dropout")

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.n_vocab, self.n_embed), # token embedding
            wpe = nn.Embedding(self.n_context, self.n_embed), # positional embedding
            drop = nn.Dropout(0.1), # dropout - may not use unless overfitting
            h = nn.ModuleList([TransformerBlock(self.config) for _ in range(self.n_layer)]),
            ln_f = nn.LayerNorm(self.n_embed)
        ))

        self.lm_head = nn.Linear(self.n_embed, self.n_vocab, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
    
    def forward(self, x, targets=None):
        B, S = x.size() # batch_size, seq_length
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        x = self.transformer.wpe(pos) + self.transformer.wte(x) 
        if self.use_dropout:
            x = self.drop(x)
        for block in self.transformer.h:
            x = block(x) # transformer blocks
        x = self.transformer.ln_f(x) # layernorm at the end
        x =  self.lm_head(x) # project embd dim to vocab dim as logits
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        return x, loss
    
    # Modified method from Andrej Kaparthy's nanoGPT implementation
    @classmethod
    def from_pretrained(cls, model_type, device=None, use_flash_attn=True):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        config_args['n_vocab'] = 50257 
        config_args['n_context'] = 1024 
        config_args['use_flash_attn'] = use_flash_attn
        # initialize model
        model = GPT(config_args, device)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a hf gpt2 model w/ language modelling head
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        print(f"Copying weights...")
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
        


        