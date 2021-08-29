

import torch.nn as nn
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


cross_entropy = nn.CrossEntropyLoss()
