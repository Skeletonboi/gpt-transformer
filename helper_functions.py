import torch
from tqdm import tqdm
from lora import LoRALayer, QLoRALayer

def generate_output(context, model, tokenizer, device, gen_length, num_samples, temp=1, top_k=50):
    """
    Sample model generated text from a given context/input.
    """
    n_context = model.config.get("n_context")
    context = tokenizer.encode(context)
    context = torch.tensor(context, dtype=torch.long).repeat(num_samples, 1)
    context = context.to(model.device)
    context_length = context.size(1)

    pbar = tqdm(total=min(context_length+gen_length, n_context), desc="Generating tokens...")
    with torch.no_grad():
        num_generated = 0
        while (context_length + num_generated < n_context) and (context_length + num_generated < gen_length):
            with torch.autocast(device_type=device, dtype=torch.bfloat16): # autocast to bfloat16 to save memory
                logits, _ = model(context)
            logits = logits[:, -1, :] / temp # adjust logits by temperature
            probs = torch.softmax(logits, dim=1) # to be consistent we softmax first
            probs, idxs = torch.topk(probs, top_k, dim=1) # [B,top_k] matrix
            tokens = torch.gather(idxs, 1, torch.multinomial(probs, num_samples=1)) # gather generated tokens

            context = torch.cat((context, tokens), dim=1) # concat to full context
            num_generated += 1
            pbar.update(1)
    pbar.close()
    return context

def replaceWithLoRA(model, replaced_modules, rank, alpha, quantize=False):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replaceWithLoRA(module, replaced_modules, rank, alpha)
        elif any(target in name for target in replaced_modules):
            if not quantize:
                lora_layer = LoRALayer(module.in_features, module.out_features, rank, alpha, \
                                       bias=module.bias is not None)
                lora_layer.linear.weight.data = module.weight.data
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data
            else:
                lora_layer = QLoRALayer(module.in_features, module.out_features, rank, alpha, \
                                        bias=module.bias is not None)
                lora_layer.linear.weight.data = module.weight.data.to(torch.int8)
                if module.bias is not None:
                    lora_layer.linear.bias.data = module.bias.data.to(torch.int8)
            setattr(model, name, lora_layer)
    return

def applyLoRA(model, lora_params):
    replaceWithLoRA(model, lora_params["replaced_modules"], lora_params["lora_rank"], lora_params["lora_alpha"], lora_params["quantize"])
    # Freeze any weights that have not been replaced with LoRA
    for name, param in model.named_parameters():
        if not any(target in name for target in lora_params["replaced_modules"]):
            param.requires_grad_(False)
    