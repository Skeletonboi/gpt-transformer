import torch
import torch.nn as nn
import tiktoken
import yaml
from tqdm import tqdm
from datasets import load_dataset
import bitsandbytes as bnb
# Inference w/ a compiled LoRA model raises torch._dynamo errors for some reason, this suppresses them
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# Custom written libraries
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
n_epoch = config["n_epoch"]
lr = float(config["lr"])
use_template = config["use_template"]
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
train_loader, test_loader = \
    SimpleDataLoader.createDataloader(dataset, batch_size, n_context, dataset_name, \
                                      tokenizer, use_template)

# INIT MODEL
if load_model:
    model = GPT(config, device)
    if lora_params["use_lora"]:
        model.unfuse()
        applyLoRA(model, lora_params)
    model.load_state_dict(torch.load(load_model_path))
else:
    model = GPT.from_pretrained(pretrained_name, device, use_sdpa=True)
    if lora_params["use_lora"]:
        model.unfuse()
        applyLoRA(model, lora_params)
model.to(device)

# COMPILE MODEL
if compile:
    print("Compiling model...")
    model = torch.compile(model)

# OPTIMIZER
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
### DEPRECATED OPTIMIZERS FOR NOW, TRAINING SEEMS UNSTABLE UNTIL I FIND OUT WHY ###
# if lora_params["quantize"]:
    # optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=lr)
# else:
#   optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)

# TRAIN
n_steps = int(train_loader.n_tokens / (batch_size * n_context))
if train:
    pbar = tqdm(total=n_steps, desc="Beginning training...")
    model.train()
    for step in range(n_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        pbar.update(1)

        # Evaluate test loss
        if step % 250 == 0:
            model.eval()
            test_loader.reset()
            with torch.no_grad():
                test_loss = 0
                for _ in range(100):
                    x, y = test_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device, dtype=torch.bfloat16):
                        _, loss = model(x, y)
                    test_loss += loss.item()
                test_loss /= 100
            model.train()
            print(f"Test Loss: {test_loss}")
    pbar.close()

# SAVE MODEL
if save_model and not load_model:
    if config["compile"]:
        torch.save(model._orig_mod.state_dict(), save_model_path)
    else:
        torch.save(model.state_dict(), save_model_path)

# GENERATE
if test_generate:
    model.eval()
    input = "If I have 2 cards in my left hand and 3 cards in my right hand, how many cards do I have in total?"
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