import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entrophy(output, target):
    return F.cross_entropy(output, target)

def F1_loss(output, target):
    return F.l1_loss(output, target)
