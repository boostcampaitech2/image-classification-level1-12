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


def f1_loss(y_true: Tensor, y_pred: Tensor, is_training=False) -> Tensor:
    '''
    https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354

    Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. 0 <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6

    '''
    assert y_true.ndim == 1
    assert y_pred.ndim == 1 or y_pred.ndim == 2

    if y_pred.ndim == 2:
        y_pred = y_pred.argmax(dim=1)

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1.requires_grad = is_training
    return f1