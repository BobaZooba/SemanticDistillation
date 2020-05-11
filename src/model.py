import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNEncoder(nn.Module):

    def __init__(self, pretrained_embeddings, rnn_dim=256, rnn_layers=2, dropout=0.3, pad_index=0):
        super().__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings=pretrained_embeddings,
                                                      padding_idx=pad_index)

        self.rnn = nn.LSTM(input_size=self.embedding.embedding_dim,
                           hidden_size=rnn_dim,
                           num_layers=rnn_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=True)

        self.out_projection = nn.Linear(in_features=rnn_dim * 2,
                                        out_features=rnn_dim)

    def forward(self, x):
        lengths = (x != self.embedding.padding_idx).sum(dim=-1)
        x = self.embedding(x)

        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.rnn(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        x = torch.max(x, dim=1)[0]
        x = self.out_projection(x)

        return x


class SimilarityWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()

        self.encoder = encoder
        self.linear = nn.Linear(1, 2)

    def forward(self, a, b):

        a = self.encoder(a)
        b = self.encoder(b)

        similarity = (a * b).sum(dim=1).unsqueeze(-1)
        logits = self.linear(similarity)

        return logits


class DistillationLoss(torch.nn.Module):

    def __init__(self, temperature=1.):
        super().__init__()

        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        teacher_prediction = torch.exp(torch.log_softmax(teacher_logits / self.temperature, dim=-1))
        student_prediction = torch.log_softmax(student_logits / self.temperature, dim=-1)

        loss = torch.mean(torch.sum(-teacher_prediction * student_prediction, dim=-1))

        return loss
