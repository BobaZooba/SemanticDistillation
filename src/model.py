import torch
from torch import nn
from torch.nn.functional import cross_entropy
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
        x = torch.nn.functional.normalize(x)

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


class EntropyLoss(nn.Module):
    """from https://arxiv.org/pdf/1902.08564.pdf"""

    def __init__(self, margin=0.):
        super().__init__()

        self.margin = margin

    def forward(self, anchors, positives):
        scores = anchors.mm(positives.t())

        margin_matrix = (torch.eye(anchors.size(0)) * self.margin).to(anchors.device)
        scores -= margin_matrix

        targets = torch.arange(anchors.size(0)).to(anchors.device)

        loss = cross_entropy(scores, targets)

        return loss


class TripletEntropyLoss(nn.Module):
    """from https://arxiv.org/pdf/1703.07737.pdf"""

    def __init__(self, delta=1e-3, nm_coefficient=0.5, np_coefficient=0.25,
                 na_coefficient=0.25, magnitude=4., margin=0.):
        super().__init__()

        self.delta = delta
        self.nm_coefficient = nm_coefficient
        self.np_coefficient = np_coefficient
        self.na_coefficient = na_coefficient
        self.magnitude = magnitude
        self.margin = margin

    def forward(self, anchors, positives, negatives):
        pos_sim = (anchors * positives).sum(-1) / self.magnitude

        neg_mul = torch.matmul(anchors, negatives.t())
        neg_mul = self.nm_coefficient * torch.exp(neg_mul / self.magnitude + self.margin)

        delta_mask = torch.eye(anchors.size(0)).bool()

        neg_pos_mul = torch.matmul(anchors, positives.t())
        neg_pos_mul = neg_pos_mul.masked_fill_(delta_mask, self.delta)
        neg_pos_mul = self.np_coefficient * torch.exp(neg_pos_mul / self.magnitude + self.margin)

        neg_anc_mul = torch.matmul(anchors, anchors.t())
        neg_anc_mul = neg_anc_mul.masked_fill_(delta_mask, self.delta)
        neg_anc_mul = self.na_coefficient * torch.exp(neg_anc_mul / self.magnitude + self.margin)

        neg_scores = torch.cat((neg_mul, neg_pos_mul, neg_anc_mul), dim=-1)

        neg_sim = torch.log(neg_scores.sum(dim=-1))

        loss = torch.relu(neg_sim - pos_sim).mean()

        return loss
