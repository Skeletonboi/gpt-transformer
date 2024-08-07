import numpy
import torch

INSTR_IN_OUT_TEMPLATE = """Below is an instruction that describes a task, paired with \
an input that provides further context. Write a response that appropriately completes the request. 

### Instruction: {}

### Input: {}

### Response: {}"""

class SimpleDataLoader:
    """
    Simple DataLoader for instruction-tuning datasets.
    Uses packed dataloading for efficient training.
    """
    def __init__(self, raw_data, batch_size, n_context, dataset_name, tokenizer, use_template=False):
        self.batch_size = batch_size
        self.n_context = n_context
        self.current_pos = 0

        self.data = []
        self.raw_data = raw_data
        self.n_samples = len(self.raw_data)
        self.eos_token = tokenizer.eot_token
        # Converting raw dataset to tokenized array
        if dataset_name == "GAIR/lima":
            self.raw_data = ["".join(sample) for sample in self.raw_data["conversations"]]
            self.data.extend(tokenizer.encode(sample) + [self.eos_token] for sample in self.raw_data)
            self.data = [token for sample in self.data for token in sample]
        elif dataset_name == "yahma/alpaca-cleaned":
            instr = self.raw_data["instruction"]
            input = self.raw_data["input"]
            output = self.raw_data["output"]
            for i in range(self.n_samples):
                if use_template:
                    sample_str = INSTR_IN_OUT_TEMPLATE.format(instr[i], input[i], output[i])
                else:
                    sample_str = "".join(instr[i]) + "".join(input[i]) + "".join(output[i])
                self.data.extend(tokenizer.encode(sample_str)+[self.eos_token])
        # Convert to tensor
        self.data = torch.tensor(self.data, dtype=int)#.to_device(device) 
        # removed to try to reduce memory usage, don't need whole dataset on device
        self.n_tokens = self.data.size()[0] # Record # tokens total in dataset, use to compute n_steps
        
    def next_batch(self):
        if self.current_pos + self.batch_size * self.n_context + 1 > self.data.size(0):
            self.current_pos = 0
        x = self.data[self.current_pos : self.current_pos + self.batch_size * self.n_context]
        y = self.data[self.current_pos + 1 : self.current_pos + self.batch_size * self.n_context + 1]
        x = x.view(self.batch_size, self.n_context)
        y = y.view(self.batch_size, self.n_context)

        self.current_pos += self.batch_size * self.n_context
        
        return x, y

    def reset(self):
        self.current_pos = 0
        return

    @classmethod
    def createDataloader(cls,  dataset, batch_size, n_context, dataset_name, tokenizer, use_template):
        # Separate train and test sets
        if dataset.get("test", None):
            raw_train = dataset["train"]
            raw_test = dataset["test"]
        else:
            train_idxs = torch.randperm(len(dataset["train"]))[:int(len(dataset["train"])*0.95)] 
            test_idxs = [i for i in range(len(dataset["train"])) if i not in train_idxs] # this is slow O(n^2) - fix
            raw_train = dataset["train"].select(train_idxs)
            raw_test = dataset["train"].select(test_idxs)
        # Create dataloaders
        import code; code.interact(local=locals())
        train_loader = SimpleDataLoader(raw_train, batch_size, n_context, dataset_name, tokenizer, use_template)
        test_loader = SimpleDataLoader(raw_test, batch_size, n_context, dataset_name, tokenizer, use_template)
        return train_loader, test_loader