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
# TOKENIZER
tokenizer = tiktoken.encoding_for_model('gpt2')
# LOAD DATA
dataset_name = "yahma/alpaca-cleaned"
dataset_name = "GAIR/lima"
dataset = load_dataset(dataset_name)
dataloader = SimpleDataLoader(config["batch_size"], config["n_context"], dataset, \
                                  dataset_name, tokenizer=tokenizer, device=device)
# TRAINING
epochs = config["n_epoch"]
lr = float(config["lr"])

model = GPT.from_pretrained('gpt2-large', device, use_flash_attn=True)
# model = GPT(config)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

for i in range(epochs):
    print(f"Training epoch {i}...")
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()

test_generate = True



if test_generate:
    model.eval()
    out = generate_output("Tell me the purpose of my existence. ", model, tokenizer, device, \
                        gen_length=100, num_samples=5, temp=1.0, top_k=50)
    for b in out:
        print("---GENERATED TEXT---")
        print(tokenizer.decode(b.tolist()))