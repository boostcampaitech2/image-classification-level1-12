import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss:
    """
    __init__(self, loss):
    loss = name of loss func, this variable can determine loss function by input string
    """

    def __init__(self, loss):
        self.loss = loss

    def loss_function(self):
        if self.loss == "":
            pass

        else:
            return nn.CrossEntropyLoss()


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
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
            reduction=self.reduction,
        )
