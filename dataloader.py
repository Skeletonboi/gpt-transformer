import numpy
import torch

class SimpleDataLoader:
    def __init__(self, batch_size, n_context, dataset, dataset_name, tokenizer, device):
        self.batch_size = batch_size
        self.n_context = n_context
        self.current_pos = 0

        self.train = ""
        self.raw_train = dataset["train"]
        self.data_size = len(self.raw_train)
        # Handling for different dataset classes from HuggingFace
        if dataset_name == "GAIR/lima":
            # for sample in self.raw_train["conversations"]:
            #     self.train += ["".join(sample)]
            self.train = "".join(["".join(sample) for sample in self.raw_train["conversations"]]) # Unsure if this is how you properly load instruction-tuning data
        elif dataset_name == "yahma/alpaca-cleaned":
            input = self.raw_train["input"]
            output = self.raw_train["output"]
            instr = self.raw_train["instruction"]
            for i in range(self.data_size):
                self.train += ["".join(instr[i]) + "".join(input[i]) + "".join(output[i])]
        # Tokenize and convert to tensor
        # self.train = [tokenizer.encode(sample) for sample in self.train]
        self.train = tokenizer.encode(self.train)
        self.train = torch.tensor(self.train, dtype=torch.bfloat16).to(device)
        
    def next_batch(self):
        return