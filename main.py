import torch
import torch.nn as nn
import tiktoken
import yaml
from tqdm import tqdm
from datasets import load_dataset
import bitsandbytes as bnb
# Custom libraries from scratch
from transformer import GPT
from dataloader import SimpleDataLoader
from helper_functions import generate_output, applyLoRA

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
compile = config["compile"]
pretrained_name = config["pretrained_name"]
save_model = config["save_model"]
save_model_path = config["save_model_path"]

load_model = config["load_model"]
load_model_path = config["load_model_path"]
lora_params = config["lora_params"]

# TOKENIZER
tokenizer = tiktoken.encoding_for_model('gpt2')

# LOAD DATA
dataset_name = "yahma/alpaca-cleaned"
# dataset_name = "GAIR/lima"
dataset = load_dataset(dataset_name)
dataloader = SimpleDataLoader(batch_size, n_context, dataset, \
                                  dataset_name, tokenizer=tokenizer, device=device)

# INIT MODEL
if load_model:
    model = GPT(config, device)
    if lora_params["use_lora"]:
        applyLoRA(model, lora_params)
    model.to(device)
    model.load_state_dict(torch.load(load_model_path))
else:
    model = GPT.from_pretrained(pretrained_name, device, use_flash_attn=True)
    if lora_params["use_lora"]:
        applyLoRA(model, lora_params)
    model.to(device)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

if lora_params["quantize"]:
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# TRAIN
if train:
    pbar = tqdm(total=n_steps, desc="Beginning training...")
    for i in range(n_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        pbar.update(1)
    pbar.close()

if save_model:
    torch.save(model._orig_mod.state_dict(), save_model_path) 

# GENERATE
if test_generate:
    model.eval()
    input = "Provide a numbered list of the 10 largest tech companies."
    out = generate_output(input, model, tokenizer, device, gen_length=250, \
                          num_samples=5, temp=1.0, top_k=50)
    for b in out:
        b = b.tolist()
        input_tk_len = len(tokenizer.encode(input))
        try:
            end = b.index(tokenizer.eot_token)
        except:
            end = len(b)
        b = b[input_tk_len:end]
        print("---PROMPT---")
        print(input)
        print("---OUTPUT---")
        print(tokenizer.decode(b))