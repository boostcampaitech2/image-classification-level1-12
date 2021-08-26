from typing import Tuple

from torch import nn, Tensor
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


# def mask_total_loss(x: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
#     mask_loss = nn.CrossEntropyLoss()(x[0], target[0])
#     gender_loss = nn.BCEWithLogitsLoss()(x[1].view(-1), target[1])
#     age_loss = nn.CrossEntropyLoss()(x[2], target[2])
#
#     total_loss = mask_loss + gender_loss + age_loss
#     return total_loss


class MaskLoss(nn.Module):
    def __init__(self, mask_weight: Tensor = None, gender_weight: Tensor = None, age_weight: Tensor = None):
        super().__init__()

        self.mask_loss_func = nn.CrossEntropyLoss(weight=mask_weight)
        self.gender_loss_func = nn.CrossEntropyLoss(weight=gender_weight)
        self.age_loss_func = nn.CrossEntropyLoss(weight=age_weight)

    def forward(self, x: Tuple[Tensor, Tensor, Tensor], target: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        mask_loss = self.mask_loss_func(x[0], target[0])
        gender_loss = self.gender_loss_func(x[1], target[1])
        age_loss = self.age_loss_func(x[2], target[2])

        total_loss = mask_loss + gender_loss + age_loss
        return total_loss
