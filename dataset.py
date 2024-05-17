import os
import torch
from torch.utils.data import Dataset


class CLSDataset(Dataset):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.samples = []
        self.load_samples(file_path)

    def load_samples(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"the file {file_path} not found")

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                label, text = line.strip().split('\t')
                if label not in ['0', '1', '2']:
                    raise ValueError(f"Invalid label: {label}")
                if len(text.strip()) == 0:
                    continue
                self.samples.append((int(label), text))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        label, text = self.samples[index]
        input_ids = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return input_ids, label_tensor


class LMDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        text = open(file_path, 'r', encoding='utf-8').read()
        self.data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, index):
        chunk = self.data[index:index+self.block_size+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
