import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional


class CosineTripletLoss(nn.Module):
    SAMPLING_TYPES = ('random', 'semi_hard', 'hard')

    def __init__(self,
                 margin: float = 0.05,
                 sampling_type: str = 'semi_hard',
                 semi_hard_margin: Optional[float] = None):
        super().__init__()

        self.margin = margin
        self.sampling_type = sampling_type
        self.semi_hard_margin = semi_hard_margin if semi_hard_margin is not None else self.margin

        if self.sampling_type not in self.SAMPLING_TYPES:
            raise ValueError(f'Not available sampling_type. Available: {", ".join(self.SAMPLING_TYPES)}')

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        positive_sim_matrix = (x * y).sum(dim=1)

        if self.sampling_type == 'random':

            batch_size = x.size(0)

            indices = torch.arange(batch_size).unsqueeze(dim=0).repeat((batch_size, 1))
            mask = ~torch.eye(batch_size).bool()
            available_indices = indices[mask].view(batch_size, batch_size - 1)
            random_indices = torch.randint(0, batch_size - 1, (batch_size,))

            negative_indices = available_indices[torch.arange(batch_size), random_indices].to(x.device)

        else:

            with torch.no_grad():
                similarity_matrix = x @ y.t()

                diagonal_mask = torch.eye(x.size(0)).bool().to(x.device)

                # positive_sim_matrix = similarity_matrix.masked_select(diagonal_mask)

                similarity_matrix = similarity_matrix.where(~diagonal_mask.bool(),
                                                            torch.tensor([-1.]).to(x.device))

                difference = positive_sim_matrix.detach().unsqueeze(-1).repeat(1, similarity_matrix.size(-1))
                difference = difference - similarity_matrix

                similarity_matrix = similarity_matrix.where(difference <= self.semi_hard_margin,
                                                            torch.tensor([-1.]).to(x.device))

                negative_indices = similarity_matrix.argmax(dim=1)

            # with torch.no_grad():
            #     similarity_matrix = x @ y.t()
            #
            #     diagonal_mask = (~torch.eye(x.size(0)).bool()).to(x.device)
            #
            #     similarity_matrix = similarity_matrix.where(~diagonal_mask.bool(),
            #                                                 torch.tensor([-1.]).to(x.device))
            #
            #     if self.sampling_type == 'semi_hard':
            #         difference = positive_sim_matrix.detach().unsqueeze(-1).repeat(1, similarity_matrix.size(-1))
            #         difference = difference - similarity_matrix
            #
            #         similarity_matrix = similarity_matrix.where(difference >= self.semi_hard_margin,
            #                                                     torch.tensor([-1.]).to(x.device))
            #
            #     negative_indices = similarity_matrix.argmax(dim=1)

        negative_sim_matrix = (x * y[negative_indices]).sum(dim=1)

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

        final_mask = target != self.ignore_index

        if mask is not None:
            final_mask = torch.logical_and(final_mask, mask)

        prediction = prediction.masked_select(final_mask)
        target = target.masked_select(final_mask)

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

    def forward(self, anchors: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:

        similarity_matrix = anchors @ positives.t()
        targets = torch.arange(anchors.size(0)).to(similarity_matrix.device)

        loss = self.criterion(similarity_matrix, targets)

        return loss
