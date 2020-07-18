import torch
from torch import nn
import torch.nn.functional as F

import transformers


class GlobalMaskedPooling(nn.Module):

    def __init__(self, pooling_type='mean', dim=1, normalize=False, length_scaling=False, square_root=False):
        super().__init__()

        self.pooling_type = pooling_type
        self.dim = dim

        self.normalize = normalize
        self.length_scaling = length_scaling
        self.square_root = square_root

        if self.pooling_type == 'max':
            self.mask_value = -100000.
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
            scaling = torch.ones(x.size(0))

        if self.length_scaling:
            lengths_factor = lengths
            if self.square_root:
                lengths_factor = lengths_factor ** 0.5
            scaling /= lengths_factor

        scaling = scaling.masked_fill(lengths == 0, 1.).unsqueeze(-1)

        if self.pooling_type == 'mean':
            x = x.mean(self.dim)
        else:
            x, _ = x.max(self.dim)

        x *= scaling

        if self.normalize:
            x = F.normalize(x)

        return x


class RawBertEncoder(nn.Module):

    def __init__(self, model_type: str):
        super().__init__()

        self.bert = transformers.AutoModel.from_pretrained(model_type)
        self.pooling = GlobalMaskedPooling(length_scaling=False, square=False)

    def forward(self, token_ids, pad_mask):
        embed = self.bert(token_ids, pad_mask)[0]

        embed = self.pooling(embed, pad_mask.bool())

        embed = F.normalize(embed)

        return embed


class BertRetrieval(nn.Module):

    def __init__(self,
                 bert_model: transformers.BertModel,
                 model_dim: int,
                 pooling_type: str = 'mean',
                 dropout_prob: float = 0.3,
                 length_scaling: bool = True,
                 square_root: bool = True):
        super().__init__()

        self.bert_model = bert_model
        self.pooling = GlobalMaskedPooling(pooling_type=pooling_type, normalize=False,
                                           length_scaling=length_scaling, square_root=square_root)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.head = nn.Linear(in_features=self.bert_model.config.hidden_size, out_features=model_dim)

        self.pad_index = self.bert_model.embeddings.word_embeddings.padding_idx

    def forward(self, x):
        pad_mask = x != self.pad_index

        x = self.bert_model(x, attention_mask=pad_mask.float())[0]

        x = self.pooling(x, pad_mask)

        x = self.dropout(x)

        x = self.head(x)

        x = F.normalize(x)

        return x
