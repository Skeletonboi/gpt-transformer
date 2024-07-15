import torch
import torch.nn as nn
from transformer import GPT
import tiktoken
import yaml
from helper_functions import generate_output
from datasets import load_dataset


with open("config.yaml") as file:
    try:
        config = yaml.safe_load(file)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

model = GPT.from_pretrained('gpt2-xl', device, use_flash_attn=True)
# model = GPT(config)
model.to(device)

tokenizer = tiktoken.encoding_for_model('gpt2')
out = generate_output("The purpose of my existence is", model, tokenizer, device, \
                      gen_length=100, num_samples=5, temp=1.0, top_k=50)
for b in out:
    print("---GENERATED TEXT---")
    print(tokenizer.decode(b.tolist()))

# dataset = load_dataset("yahma/alpaca-cleaned")
# training code
# model.train()

