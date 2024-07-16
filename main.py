import torch
import torch.nn as nn
import tiktoken
import yaml
from datasets import load_dataset
# Custom libraries from scratch
from transformer import GPT
from dataloader import SimpleDataLoader
from helper_functions import generate_output

# LOAD CONFIG
with open("config.yaml") as file:
    try:
        config = yaml.safe_load(file)
        print(config)
    except yaml.YAMLError as exc:
        print(exc)

# SET DEVICE
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

# CONFIG PARAMS
n_steps = config["n_steps"]
lr = float(config["lr"])
test_generate = config["test_generate"]
batch_size = config["batch_size"]
n_context = config["n_context"]
train = config["train"]

# TOKENIZER
tokenizer = tiktoken.encoding_for_model('gpt2')

# LOAD DATA
dataset_name = "yahma/alpaca-cleaned"
dataset_name = "GAIR/lima"
dataset = load_dataset(dataset_name)
dataloader = SimpleDataLoader(batch_size, n_context, dataset, \
                                  dataset_name, tokenizer=tokenizer, device=device)

# INIT MODEL
model = GPT.from_pretrained('gpt2-large', device, use_flash_attn=True)
# model = GPT(config)
model.to(device)
model = torch.compile(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# TRAIN
if train:
    for i in range(n_steps):
        print(f"Training step {i}...")
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), "./model.pth") 

# GENERATE
if test_generate:
    model.eval()
    out = generate_output("Tell me the purpose of my existence. ", model, tokenizer, device, \
                        gen_length=100, num_samples=5, temp=1.0, top_k=50)
    for b in out:
        print("---GENERATED TEXT---")
        print(tokenizer.decode(b.tolist()))