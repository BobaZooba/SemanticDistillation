import math
import json

import numpy as np
from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import transformers
from tqdm import tqdm


class GlobalMaskedPooling(nn.Module):

    def __init__(self, pooling_type='mean', dim=1, length_scaling=True, square=True):
        super().__init__()

        self.pooling_type = pooling_type
        self.dim = dim
        self.length_scaling = length_scaling
        self.square = square

        if self.pooling_type == 'max':
            self.mask_value = -10000.
        else:
            self.mask_value = 0.

        if self.pooling_type not in ['mean', 'max']:
            raise ValueError('Available types: mean, max')

    def forward(self, x, pad_mask):
        lengths = pad_mask.sum(self.dim).float()

        x = x.masked_fill((~pad_mask).unsqueeze(-1), self.mask_value)

        if self.pooling_type == 'mean':
            scaling = x.size(self.dim) / lengths
        else:
            scaling = torch.ones(x.size(self.dim))

        if self.length_scaling:
            lengths_factor = lengths
            if self.square:
                lengths_factor = lengths_factor ** 0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.).unsqueeze(-1)

        if self.pooling_type == 'mean':
            x = x.mean(self.dim)
        else:
            x = x.max(self.dim)

        x *= scaling

        return x

    def extra_repr(self) -> str:
        return f'pooling_type="{self.pooling_type}"'


class Encoder(nn.Module):

    def __init__(self, model_type: str):
        super().__init__()

        self.bert = transformers.AutoModel.from_pretrained(model_type)
        self.pooling = GlobalMaskedPooling(length_scaling=False, square=False)

    def forward(self, token_ids, pad_mask):
        embed = self.bert(token_ids, pad_mask)[0]

        embed = self.pooling(embed, pad_mask.bool())

        embed = F.normalize(embed)

        return embed
