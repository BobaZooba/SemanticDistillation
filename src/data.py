import torch
from torch.utils.data import Dataset


class PairedData(Dataset):

    def __init__(self, word2index, texts_a, texts_b, targets, logits=None, pad_index=0, max_length=64):

        self.word2index = word2index
        self.pad_index = pad_index
        self.max_length = max_length

        self.texts_a = [self.padding(self.indexing(sample)) for sample in texts_a]
        self.texts_b = [self.padding(self.indexing(sample)) for sample in texts_b]
        self.targets = targets
        self.logits = logits

    def indexing(self, tokens):
        return [self.word2index[tok] for tok in tokens if tok in self.word2index]

    def padding(self, sequence, max_length=None):

        max_length = min(self.max_length, max_length) if max_length is not None else self.max_length

        sequence = sequence[:max_length]

        sequence += [self.pad_index] * (max_length - len(sequence))

        return sequence

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        text_a = torch.tensor(self.texts_a[index])
        text_b = torch.tensor(self.texts_b[index])
        target = self.targets[index]

        if self.logits is not None:
            logits = self.logits[index]
            return text_a, text_b, target, logits
        else:
            return text_a, text_b, target
