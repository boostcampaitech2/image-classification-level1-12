import torch.nn.functional as F
from torch import torch, nn, Tensor
from typing import Tuple


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy(output, target):
    return F.cross_entropy(output, target)

def l1_loss(output, target):
    return F.l1_loss(output, F.one_hot(target, output.shape[1]))

class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.total_loss = nn.CrossEntropyLoss()
        self.gender_loss_func = nn.BCEWithLogitsLoss()
        self.age_loss_func = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return self.total_loss(x, target)


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from kornia.utils.one_hot import one_hot

# # based on:
# # https://github.com/zhezh/focalloss/blob/master/focalloss.py


# def focal_loss(
#     input: torch.Tensor,
#     target: torch.Tensor,
#     alpha: float,
#     gamma: float = 2.0,
#     reduction: str = 'none',
#     eps: float = 1e-8,
# ) -> torch.Tensor:
#     r"""Criterion that computes Focal loss.
#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:
#     .. math::
#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.
#     Args:
#         input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
#         target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
#         alpha: Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma: Focusing parameter :math:`\gamma >= 0`.
#         reduction: Specifies the reduction to apply to the
#           output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
#           will be applied, ``'mean'``: the sum of the output will be divided by
#           the number of elements in the output, ``'sum'``: the output will be
#           summed.
#         eps: Scalar to enforce numerical stabiliy.
#     Return:
#         the computed loss.
#     Example:
#         >>> N = 5  # num_classes
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
#         >>> output.backward()
#     """
#     if not isinstance(input, torch.Tensor):
#         raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

#     if not len(input.shape) >= 2:
#         raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

#     if input.size(0) != target.size(0):
#         raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

#     n = input.size(0)
#     out_size = (n,) + input.size()[2:]
#     if target.size()[1:] != input.size()[2:]:
#         raise ValueError(f'Expected target size {out_size}, got {target.size()}')

#     if not input.device == target.device:
#         raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

#     # compute softmax over the classes axis
#     input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

#     # create the labels one hot tensor
#     target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

#     # compute the actual focal loss
#     weight = torch.pow(-input_soft + 1.0, gamma)

#     focal = -alpha * weight * torch.log(input_soft)
#     loss_tmp = torch.sum(target_one_hot * focal, dim=1)

#     if reduction == 'none':
#         loss = loss_tmp
#     elif reduction == 'mean':
#         loss = torch.mean(loss_tmp)
#     elif reduction == 'sum':
#         loss = torch.sum(loss_tmp)
#     else:
#         raise NotImplementedError(f"Invalid reduction mode: {reduction}")
#     return loss


# class FocalLoss(nn.Module):
#     r"""Criterion that computes Focal loss.
#     According to :cite:`lin2018focal`, the Focal loss is computed as follows:
#     .. math::
#         \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
#     Where:
#        - :math:`p_t` is the model's estimated probability for each class.
#     Args:
#         alpha: Weighting factor :math:`\alpha \in [0, 1]`.
#         gamma: Focusing parameter :math:`\gamma >= 0`.
#         reduction: Specifies the reduction to apply to the
#           output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
#           will be applied, ``'mean'``: the sum of the output will be divided by
#           the number of elements in the output, ``'sum'``: the output will be
#           summed.
#         eps: Scalar to enforce numerical stabiliy.
#     Shape:
#         - Input: :math:`(N, C, *)` where C = number of classes.
#         - Target: :math:`(N, *)` where each value is
#           :math:`0 ≤ targets[i] ≤ C−1`.
#     Example:
#         >>> N = 5  # num_classes
#         >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
#         >>> criterion = FocalLoss(**kwargs)
#         >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
#         >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
#         >>> output = criterion(input, target)
#         >>> output.backward()
#     """

#     def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: float = 1e-8) -> None:
#         super().__init__()
#         self.alpha: float = alpha
#         self.gamma: float = gamma
#         self.reduction: str = reduction
#         self.eps: float = eps

#     def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#         return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps)