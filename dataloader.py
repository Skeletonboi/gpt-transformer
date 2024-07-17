import numpy
import torch

class SimpleDataLoader:
    """
    Simple DataLoader for instruction-tuning datasets.
    Uses packed dataloading for efficient training.
    """
    def __init__(self, batch_size, n_context, dataset, dataset_name, tokenizer, device):
        self.batch_size = batch_size
        self.n_context = n_context
        self.current_pos = 0

        self.train = []
        self.raw_train = dataset["train"]
        self.n_samples = len(self.raw_train)
        self.eos_token = tokenizer.eot_token
        # Handling for different dataset classes from HuggingFace
        if dataset_name == "GAIR/lima":
            self.raw_train = ["".join(sample) for sample in self.raw_train["conversations"]]
            self.train.extend(tokenizer.encode(sample) + [self.eos_token] for sample in self.raw_train)
            self.train = [token for sample in self.train for token in sample]
        elif dataset_name == "yahma/alpaca-cleaned":
            input = self.raw_train["input"]
            output = self.raw_train["output"]
            instr = self.raw_train["instruction"]
            for i in range(self.n_samples):
                sample_str = "".join(instr[i]) + "".join(input[i]) + "".join(output[i])
                self.train.extend(tokenizer.encode(sample_str)+[self.eos_token])

        # Convert to tensor
        self.train = torch.tensor(self.train, dtype=int)#.to_device(device) 
        # removed to try to reduce memory usage, don't need whole dataset on device
        
    def next_batch(self):
        if self.current_pos + self.batch_size * self.n_context + 1 > self.train.size(0):
            self.current_pos = 0
        x = self.train[self.current_pos : self.current_pos + self.batch_size * self.n_context]
        y = self.train[self.current_pos + 1 : self.current_pos + self.batch_size * self.n_context + 1]
        x = x.view(self.batch_size, self.n_context)
        y = y.view(self.batch_size, self.n_context)

        self.current_pos += self.batch_size * self.n_context
        
        return x, y