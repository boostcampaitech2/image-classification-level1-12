import torch.nn.functional as F
from torch import torch, nn, Tensor
from typing import Tuple

# criterion = nn.BCEWithLogitsLoss()  #시그모이드가 로스에 추가됨

class MaskLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.total_loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor, target: Tensor) -> Tensor:
        return self.total_loss(x, target)


# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )