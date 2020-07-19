import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from abc import ABC


class CosineMiner(nn.Module, ABC):
    SAMPLING_TYPES = ('random', 'semi_hard', 'hard')

    def __init__(self,
                 n_negatives: int = 1,
                 sampling_type: str = 'semi_hard',
                 normalize: bool = False,
                 multinomial: bool = False,
                 semi_hard_epsilon: float = 0.,
                 margin: Optional[float] = None,
                 mask_value: float = -10.):
        super().__init__()

        self.n_negatives = n_negatives
        self.sampling_type = sampling_type
        self.normalize = normalize
        self.multinomial = multinomial
        self.margin = margin
        self.semi_hard_epsilon = semi_hard_epsilon

        if self.sampling_type not in self.SAMPLING_TYPES:
            raise ValueError(f'Not available sampling_type. Available: {", ".join(self.SAMPLING_TYPES)}')

        self.mask_value = torch.tensor([mask_value])
        self.diagonal_mask_value = torch.tensor([-10000.])

    def get_indices(self, similarity_matrix):
        if self.multinomial:
            similarity_matrix = torch.softmax(similarity_matrix, dim=1)
            negative_indices = torch.multinomial(similarity_matrix, num_samples=self.n_negatives)
        else:
            negative_indices = similarity_matrix.argsort(descending=True)
            negative_indices = negative_indices[:, :self.n_negatives]
        return negative_indices

    def random_sampling(self, batch_size: int) -> torch.Tensor:
        possible_indices = torch.arange(batch_size).unsqueeze(dim=0).repeat(batch_size, 1)
        mask = ~torch.eye(batch_size).bool()
        possible_indices = possible_indices.masked_select(mask).view(batch_size, batch_size - 1)
        random_indices = torch.randint(batch_size - 1, (batch_size, self.n_negatives))
        negative_indices = torch.gather(possible_indices, 1, random_indices)

        return negative_indices

    def semi_hard_sampling(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            if self.normalize:
                anchor = F.normalize(anchor)
                positive = F.normalize(positive)

            similarity_matrix = anchor @ positive.t()

            diagonal_mask = torch.eye(anchor.size(0)).bool().to(anchor.device)
            positive_sim_matrix = similarity_matrix.masked_select(diagonal_mask)

            difference = positive_sim_matrix.detach().unsqueeze(-1).repeat(1, similarity_matrix.size(-1))
            difference = difference - similarity_matrix + self.semi_hard_epsilon

            similarity_matrix = similarity_matrix.where(~diagonal_mask.bool(),
                                                        self.diagonal_mask_value.to(anchor.device))

            similarity_matrix = similarity_matrix.where(difference > 0.,
                                                        self.mask_value.to(anchor.device))

            if self.margin is not None:
                similarity_matrix = similarity_matrix.where(difference <= self.margin,
                                                            self.mask_value.to(anchor.device))

            negative_indices = self.get_indices(similarity_matrix)

        return negative_indices

    def hard_sampling(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():

            if self.normalize:
                anchor = F.normalize(anchor)
                positive = F.normalize(positive)

            similarity_matrix = anchor @ positive.t()

            diagonal_mask = torch.eye(anchor.size(0)).bool().to(anchor.device)

            similarity_matrix = similarity_matrix.where(~diagonal_mask.bool(),
                                                        self.diagonal_mask_value.to(anchor.device))

            negative_indices = self.get_indices(similarity_matrix)

        return negative_indices

    def sampling(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:

        if self.sampling_type == 'hard':
            negative_indices = self.hard_sampling(anchor=anchor, positive=positive)
        elif self.sampling_type == 'semi_hard':
            negative_indices = self.semi_hard_sampling(anchor=anchor, positive=positive)
        else:
            negative_indices = self.random_sampling(batch_size=anchor.size(0))

        negative_indices = negative_indices.to(anchor.device)

        return negative_indices


class CosineTripletLoss(nn.Module):

    def __init__(self,
                 margin: float = 0.05,
                 n_negatives: int = 5,
                 sampling_type: str = 'semi_hard',
                 normalize: bool = False,
                 multinomial: bool = False,
                 use_margin_for_sampling: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__()

        self.margin = margin

        self.miner = CosineMiner(n_negatives=n_negatives,
                                 sampling_type=sampling_type,
                                 normalize=normalize,
                                 multinomial=multinomial,
                                 margin=self.margin if use_margin_for_sampling else None,
                                 semi_hard_epsilon=semi_hard_epsilon)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:

        positive_sim_matrix = (anchor * positive).sum(dim=1)

        negative_indices = self.miner.sampling(anchor=anchor, positive=positive)

        negative = positive[negative_indices]

        if negative_indices.size(1) == 1:
            negative = negative.squeeze(dim=1)
        else:
            negative = negative.mean(dim=1)

        negative_sim_matrix = (anchor * negative).sum(dim=1)

        loss = torch.relu(self.margin - positive_sim_matrix + negative_sim_matrix).mean()

        return loss


class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing: float = 0.1, use_kl: bool = False, ignore_index: int = -100):
        super().__init__()

        assert 0 <= smoothing < 1

        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.use_kl = use_kl

    def smooth_one_hot(self, true_labels: torch.Tensor, classes: int) -> torch.Tensor:

        confidence = 1.0 - self.smoothing

        with torch.no_grad():
            true_dist = torch.empty(size=(true_labels.size(0), classes), device=true_labels.device)
            true_dist.fill_(self.smoothing / (classes - 1))
            true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)

        return true_dist

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param prediction: [batch_size, num_classes]
        :param target: [batch_size]
        :param mask: [batch_size, num_classes] True if need
        :return: scalar
        """

        # final_mask = target != self.ignore_index
        #
        # if mask is not None:
        #     final_mask = torch.logical_and(final_mask, mask)
        #
        # prediction = prediction.masked_select(final_mask)
        # target = target.masked_select(final_mask)

        prediction = F.log_softmax(prediction, dim=-1)

        target_smoothed_dist = self.smooth_one_hot(target, classes=prediction.size(-1))

        if self.use_kl:
            loss = F.kl_div(prediction, target_smoothed_dist, reduction='batchmean')
        else:
            loss = torch.mean(torch.sum(-target_smoothed_dist * prediction, dim=-1))

        return loss


class MultipleNegativesLoss(nn.Module):

    def __init__(self, smoothing: float = 0.1):
        super().__init__()

        if smoothing == 0.:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = LabelSmoothingLoss(smoothing=smoothing)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:

        similarity_matrix = anchor @ positive.t()
        targets = torch.arange(anchor.size(0)).to(similarity_matrix.device)

        loss = self.criterion(similarity_matrix, targets)

        return loss


class MultipleNegativesWithMiningLoss(MultipleNegativesLoss):

    def __init__(self,
                 smoothing: float = 0.1,
                 n_negatives: int = 4,
                 miner_type: str = 'cosine',
                 sampling_type: str = 'semi_hard',
                 normalize: bool = True,
                 multinomial: bool = False,
                 semi_hard_epsilon: float = 0.):
        super().__init__(smoothing=smoothing)

        if miner_type == 'cosine':
            self.miner = CosineMiner(n_negatives=n_negatives,
                                     sampling_type=sampling_type,
                                     normalize=normalize,
                                     multinomial=multinomial,
                                     semi_hard_epsilon=semi_hard_epsilon)
        else:
            raise ValueError('Not available miner_type')

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:

        negative_indices = self.miner.sampling(anchor=anchor, positive=positive)

        negative = positive[negative_indices]

        candidates = torch.cat((positive.unsqueeze(dim=1), negative), dim=1)

        anchor = anchor.unsqueeze(dim=1)

        similarity_matrix = torch.bmm(anchor, candidates.transpose(1, 2)).squeeze(dim=1)

        target = torch.zeros(anchor.size(0)).long().to(similarity_matrix.device)

        loss = self.criterion(similarity_matrix, target)

        return loss
